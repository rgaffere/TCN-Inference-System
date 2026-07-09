import json
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =========================
# Model config
# =========================

T = 512
IN_CHANNELS = 6
OUT_CHANNELS = 6
CPL = 16
LAYER_COUNT = 7
K = 3
DILATIONS = [1, 2, 4, 8, 16, 32, 64]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CUDA runs as reproducible as practical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def receptive_field():
    return 1 + 2 * (K - 1) * sum(DILATIONS)


# =========================
# Dataset
# =========================

class MultiFileIMUDataset(torch.utils.data.Dataset):
    def __init__(self, arrays):
        self.arrays = [a.astype(np.float32) for a in arrays]
        self.index = []

        for file_idx, data in enumerate(self.arrays):
            n = len(data) - T
            for start_idx in range(n):
                self.index.append((file_idx, start_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start_idx = self.index[idx]
        data = self.arrays[file_idx]

        x = data[start_idx : start_idx + T]
        y = data[start_idx + T]

        # from_numpy avoids an extra copy versus torch.tensor().
        # np.ascontiguousarray makes x.T safe/fast for PyTorch batching.
        return torch.from_numpy(np.ascontiguousarray(x.T)), torch.from_numpy(y)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, groups=1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            dilation=dilation,
            padding=0,
            groups=groups
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, use_projection, depthwise_conv1):
        super().__init__()

        conv1_groups = in_ch if depthwise_conv1 else 1

        self.conv1 = CausalConv1d(
            in_ch,
            out_ch,
            K,
            dilation,
            groups=conv1_groups
        )

        # Conv2 is always depthwise: one temporal kernel per output channel.
        self.conv2 = CausalConv1d(
            out_ch,
            out_ch,
            K,
            dilation,
            groups=out_ch
        )

        if use_projection:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        out = self.conv1(x)
        out = torch.relu(out)

        out = self.conv2(out)
        out = torch.relu(out)

        out = out + res

        return out


class StreamingTCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = ResidualBlock(
            in_ch=IN_CHANNELS,
            out_ch=CPL,
            dilation=DILATIONS[0],
            use_projection=True,
            depthwise_conv1=False
        )

        self.hidden_layers = nn.ModuleList([
            ResidualBlock(
                in_ch=CPL,
                out_ch=CPL,
                dilation=DILATIONS[i],
                use_projection=False,
                depthwise_conv1=True
            )
            for i in range(1, LAYER_COUNT)
        ])

        self.output_layer = nn.Conv1d(CPL, OUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        y = self.output_layer(x)

        # latest point only, matching doOutputLayer()
        return y[:, :, -1]


# =========================
# Export
# =========================

def export_for_cpp(model, mean, std, path):
    m = model

    export = {
        "T": T,
        "inChannels": IN_CHANNELS,
        "layerCount": LAYER_COUNT,
        "cpl": CPL,
        "k": K,
        "dilations": DILATIONS,
        "outChannels": OUT_CHANNELS,

        "mean": mean.tolist(),
        "std": std.tolist(),

        "inputFilter1": m.input_layer.conv1.conv.weight.detach().cpu().numpy().tolist(),
        "inputBias1": m.input_layer.conv1.conv.bias.detach().cpu().numpy().tolist(),

        "inputFilter2": m.input_layer.conv2.conv.weight.detach().cpu().numpy()[:, 0, :].tolist(),
        "inputBias2": m.input_layer.conv2.conv.bias.detach().cpu().numpy().tolist(),

        "inputResidualFilter": m.input_layer.residual.weight.detach().cpu().numpy()[:, :, 0].tolist(),
        "inputResidualBias": m.input_layer.residual.bias.detach().cpu().numpy().tolist(),

        "hiddenFilter1": [],
        "hiddenBias1": [],
        "hiddenFilter2": [],
        "hiddenBias2": [],

        "outputFilter": m.output_layer.weight.detach().cpu().numpy()[:, :, 0].tolist(),
        "outputBiases": m.output_layer.bias.detach().cpu().numpy().tolist()
    }

    for layer in m.hidden_layers:
        export["hiddenFilter1"].append(
            layer.conv1.conv.weight.detach().cpu().numpy()[:, 0, :].tolist()
        )
        export["hiddenBias1"].append(
            layer.conv1.conv.bias.detach().cpu().numpy().tolist()
        )
        export["hiddenFilter2"].append(
            layer.conv2.conv.weight.detach().cpu().numpy()[:, 0, :].tolist()
        )
        export["hiddenBias2"].append(
            layer.conv2.conv.bias.detach().cpu().numpy().tolist()
        )

    with open(path, "w") as f:
        json.dump(export, f, indent=2)

# =========================
# Training
# =========================

def load_imu_csv(path):
    df = pd.read_csv(path, header=None)

    # directly select by column index
    data = df.iloc[:, [10, 11, 12, 4, 5, 6]].values.astype(np.float32)

    return data



def split_train_val_arrays(arrays, split_mode="time", val_fraction=0.2, val_file_idx=-1):
    """
    Returns train_arrays_raw, val_arrays_raw.

    split_mode="random_window":
        Randomly split windows across all files. Recommended for stable
        architecture comparisons when there are only a few recordings.

    split_mode="time":
        Split each recording by time. Every CSV contributes to train and val.
        This avoids a single held-out file dominating validation when only a few
        recordings are available.

    split_mode="file":
        Original behavior. First (1-val_fraction) files train, remaining files val.

    split_mode="lofo":
        Leave-one-file-out. Train on all files except val_file_idx and validate on
        that one file. Use this for strict unseen-recording evaluation.
    """
    if len(arrays) == 0:
        raise ValueError("No input arrays were loaded.")

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    if split_mode == "time":
        train_arrays_raw = []
        val_arrays_raw = []

        for file_idx, a in enumerate(arrays):
            split = int(len(a) * (1.0 - val_fraction))

            # Need at least T history samples plus one prediction target in each split.
            if split <= T:
                raise ValueError(
                    f"File {file_idx} is too short for the requested time split: "
                    f"train length would be {split}, T={T}."
                )
            if len(a) - split <= T:
                raise ValueError(
                    f"File {file_idx} is too short for the requested time split: "
                    f"validation length would be {len(a) - split}, T={T}. "
                    f"Use a smaller --val_fraction or longer recordings."
                )

            train_arrays_raw.append(a[:split])
            val_arrays_raw.append(a[split:])

        print(f"Split mode: time-based within each file ({100*(1-val_fraction):.1f}% train / {100*val_fraction:.1f}% val)")
        return train_arrays_raw, val_arrays_raw

    if split_mode == "file":
        split = int(len(arrays) * (1.0 - val_fraction))
        split = max(1, min(split, len(arrays) - 1))
        print(f"Split mode: file-level split ({split} train files / {len(arrays) - split} val files)")
        return arrays[:split], arrays[split:]

    if split_mode == "lofo":
        if val_file_idx < 0:
            val_file_idx = len(arrays) - 1
        if val_file_idx >= len(arrays):
            raise ValueError(f"val_file_idx={val_file_idx} is out of range for {len(arrays)} files")

        train_arrays_raw = [a for i, a in enumerate(arrays) if i != val_file_idx]
        val_arrays_raw = [arrays[val_file_idx]]
        print(f"Split mode: leave-one-file-out (validation file index {val_file_idx})")
        return train_arrays_raw, val_arrays_raw

    raise ValueError(f"Unknown split_mode: {split_mode}")




def build_purged_random_window_split(arrays, val_fraction, seed, block_size, purge_gap):
    """
    Build a randomized window-level validation split without adjacent-window leakage.

    Windows are first grouped into contiguous blocks within each file. Random blocks are
    assigned to validation until approximately val_fraction of windows are held out.
    Training windows are then removed if their start index is within purge_gap samples
    of any validation block in the same file.

    This keeps validation coverage broad across all recordings while preventing nearly
    identical neighboring windows from appearing in both train and validation.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be nonnegative, got {purge_gap}")

    window_counts = []
    offsets = []
    total = 0
    for file_idx, a in enumerate(arrays):
        n = len(a) - T
        if n <= 0:
            raise ValueError(f"File {file_idx} is too short: len={len(a)}, T={T}")
        offsets.append(total)
        window_counts.append(n)
        total += n

    blocks = []
    for file_idx, n in enumerate(window_counts):
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            blocks.append((file_idx, start, end, end - start))

    rng = random.Random(seed if seed is not None else 0)
    rng.shuffle(blocks)

    target_val = max(1, int(round(total * val_fraction)))
    val_blocks = []
    val_windows = 0
    for block in blocks:
        if val_windows >= target_val:
            break
        val_blocks.append(block)
        val_windows += block[3]

    val_masks = [np.zeros(n, dtype=bool) for n in window_counts]
    excluded_train_masks = [np.zeros(n, dtype=bool) for n in window_counts]

    for file_idx, start, end, _ in val_blocks:
        val_masks[file_idx][start:end] = True
        purge_start = max(0, start - purge_gap)
        purge_end = min(window_counts[file_idx], end + purge_gap)
        excluded_train_masks[file_idx][purge_start:purge_end] = True

    train_indices = []
    val_indices = []
    train_window_masks = []

    for file_idx, n in enumerate(window_counts):
        train_mask = ~excluded_train_masks[file_idx]
        train_window_masks.append(train_mask)

        offset = offsets[file_idx]
        train_starts = np.flatnonzero(train_mask)
        val_starts = np.flatnonzero(val_masks[file_idx])

        train_indices.extend((offset + train_starts).tolist())
        val_indices.extend((offset + val_starts).tolist())

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(
            f"Invalid purged split: train windows={len(train_indices)}, val windows={len(val_indices)}. "
            f"Try smaller --val_fraction, smaller --purge_gap, or larger --block_size."
        )

    return train_indices, val_indices, train_window_masks, total


def compute_train_mean_std_from_window_masks(arrays, train_window_masks):
    """Compute normalization stats using only samples covered by training windows."""
    covered_chunks = []

    for file_idx, (a, mask) in enumerate(zip(arrays, train_window_masks)):
        n_windows = len(mask)
        diff = np.zeros(len(a) + 1, dtype=np.int32)
        train_starts = np.flatnonzero(mask)

        for s in train_starts:
            diff[s] += 1
            end = min(len(a), s + T + 1)  # input history plus prediction target
            diff[end] -= 1

        covered = np.cumsum(diff[:-1]) > 0
        if not np.any(covered):
            raise ValueError(f"No training samples covered in file {file_idx}")
        covered_chunks.append(a[covered])

    train_data = np.concatenate(covered_chunks, axis=0)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8
    return mean, std


def train(args):
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: CUDA not available. Training on CPU.")

    arrays = [load_imu_csv(path) for path in args.csv]

    rf = receptive_field()
    print(f"Receptive field: {rf} samples")

    if args.split_mode == "random_window":
        train_indices, val_indices, train_window_masks, total_windows = build_purged_random_window_split(
            arrays=arrays,
            val_fraction=args.val_fraction,
            seed=args.seed,
            block_size=args.block_size,
            purge_gap=args.purge_gap,
        )

        mean, std = compute_train_mean_std_from_window_masks(arrays, train_window_masks)

        norm_arrays = [
            (a - mean) / std
            for a in arrays
        ]

        full_set = MultiFileIMUDataset(norm_arrays)
        train_set = torch.utils.data.Subset(full_set, train_indices)
        val_set = torch.utils.data.Subset(full_set, val_indices)

        print(
            f"Split mode: purged random-window block split "
            f"(target {100*(1-args.val_fraction):.1f}% train / {100*args.val_fraction:.1f}% val)"
        )
        print(f"Total windows before purge: {total_windows}")
        print(f"Train windows after purge: {len(train_set)}")
        print(f"Validation windows: {len(val_set)}")
        print(f"Block size: {args.block_size} windows")
        print(f"Purge gap: {args.purge_gap} samples/windows around validation blocks")
    else:
        train_arrays_raw, val_arrays_raw = split_train_val_arrays(
            arrays,
            split_mode=args.split_mode,
            val_fraction=args.val_fraction,
            val_file_idx=args.val_file_idx
        )

        all_train_data = np.concatenate(
            train_arrays_raw,
            axis=0
        )

        mean = all_train_data.mean(axis=0)
        std = all_train_data.std(axis=0) + 1e-8

        train_arrays = [
            (a - mean) / std
            for a in train_arrays_raw
        ]

        val_arrays = [
            (a - mean) / std
            for a in val_arrays_raw
        ]

        train_set = MultiFileIMUDataset(train_arrays)
        val_set = MultiFileIMUDataset(val_arrays)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    use_cuda = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0)
    )

    model = StreamingTCN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = math.inf

    patience = 8
    bad_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                pred = model(x)
                loss = loss_fn(pred, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), args.model_out)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation MSE: {best_val:.6f}")

    model.load_state_dict(torch.load(args.model_out, map_location=DEVICE))
    export_for_cpp(model, mean, std, args.export_json)

    plot_validation(model, val_set, mean, std, args.plot_out)


# =========================
# Plot validation
# =========================

def plot_validation(model, val_set, mean, std, plot_path):
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for i in range(min(1000, len(val_set))):
            x, y = val_set[i]
            x = x.unsqueeze(0).to(DEVICE)

            pred = model(x).cpu().numpy()[0]

            preds.append(pred)
            actuals.append(y.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    preds_denorm = preds * std + mean
    actuals_denorm = actuals * std + mean

    labels = [
        "user_acc_x(G)", "user_acc_y(G)", "user_acc_z(G)",
        "rotation_rate_x(radians_s)", "rotation_rate_y(radians_s)", "rotation_rate_z(radians_s)"
    ]

    for ch in range(6):
        plt.figure()
        plt.plot(actuals_denorm[:, ch], label="Actual")
        plt.plot(preds_denorm[:, ch], label="Predicted")
        plt.title(f"TCN Prediction Validation: {labels[ch]}")
        plt.xlabel("Sample")
        plt.ylabel(labels[ch])
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_path}_{labels[ch]}.png", dpi=200)
        plt.close()

    error = np.mean((preds_denorm - actuals_denorm) ** 2, axis=1)

    plt.figure()
    plt.plot(error)
    plt.title("Prediction Error Over Time")
    plt.xlabel("Sample")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig(f"{plot_path}_error.png", dpi=200)
    plt.close()

    print(f"Saved validation plots as {plot_path}_*.png")


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, nargs="+")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_mode", choices=["random_window", "time", "file", "lofo"], default="random_window")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--val_file_idx", type=int, default=-1)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--purge_gap", type=int, default=T)

    parser.add_argument("--model_out", default="tcn_imu_model_depthwise.pt")
    parser.add_argument("--export_json", default="tcn_imu_weights_depthwise.json")
    parser.add_argument("--plot_out", default="validation")

    args = parser.parse_args()

    train(args)
# Real-Time Predictive Signal Hardware

End-to-end hardware/software co-design of an **INT8 Temporal Convolutional Network (TCN) accelerator** for real-time multichannel signal prediction and anomaly detection.

The project spans the complete development flow from **PyTorch model training** and **C++ reference inference** through **SystemVerilog RTL**, **AXI4-Stream integration**, **randomized verification**, **Synopsys VC SpyGlass analysis**, and **Synopsys Design Compiler synthesis and timing closure** using the **SkyWater SKY130 PDK**.

> **Project status: Complete**  
> Developed independently from April 2026 through July 2026.

---

## Headline Results

| Result | Value |
|---|---:|
| End-to-end accelerator throughput | **2.02 GMAC/s** |
| End-to-end inference latency | **0.94 µs** |
| Integrated accelerator clock | **170 MHz** |
| Randomized verification scenarios | **3,000+** |
| Synopsys Design Compiler timing violations | **0** |
| Synopsys VC SpyGlass fatal issues / errors | **0 / 0** |
| TCN receptive field | **509 samples** |
| Numeric precision | **INT8 weights and activations, 32-bit accumulators** |

The design targets streaming, low-latency edge inference rather than batch processing. It accepts multichannel samples through an **AXI4-Stream** interface, maintains causal history in **SRAM-backed ring buffers**, evaluates the TCN in hardware, and produces predicted output samples for downstream anomaly detection.

---

## What This Project Demonstrates

- Parameterized **SystemVerilog RTL** design for a complete neural-network accelerator
- Pipelined **multiply-accumulate datapaths** and fixed-point arithmetic
- **INT8 quantization**, ReLU, saturation, residual connections, and causal convolution
- Integration of synchronous **SKY130 SRAM macros** into a streaming architecture
- **AXI4-Stream** valid/ready handshaking and backpressure-aware control
- Self-checking **SystemVerilog verification** with randomized stimulus, scoreboards, assertions, and golden-model comparison
- RTL lint and static analysis with **Synopsys VC SpyGlass**
- ASIC synthesis, PPA analysis, and timing closure with **Synopsys Design Compiler Ultra**
- Cross-validation among **PyTorch**, **C++**, and RTL implementations
- Architecture-level comparison of depthwise and channel-mixing residual blocks

---

## Network Architecture

The implemented TCN processes six-channel inertial measurement unit data and predicts the next multichannel signal sample.

- **Input channels:** 6
- **Hidden channels:** 16
- **Output channels:** 6
- **Residual blocks:** 7
- **Kernel size:** 3
- **Dilations:** 1, 2, 4, 8, 16, 32, 64
- **Receptive field:** 509 samples
- **Weights and activations:** signed INT8
- **Accumulators:** signed 32-bit

Each residual block contains two causal convolution stages followed by fixed-point requantization and ReLU. The second stage is combined with the delayed residual input using saturated signed arithmetic.

The implementation is parameterized so channel counts, dilation, kernel dimensions, and datapath widths can be adapted without rewriting the core modules.

---

## RTL Architecture

### Streaming Datapath

The accelerator is organized as reusable input, hidden, output, and anomaly-detection blocks. Pipeline control propagates valid data through the architecture while preserving cycle alignment between the convolution path and residual path.

A **16-lane MAC array** computes all hidden output channels in parallel. Signed INT8 operands are accumulated in 32-bit registers before requantization back to INT8.

The datapath implements:

- Dilated causal convolution
- Parallel multiply-accumulate operations
- Fixed-point requantization
- ReLU activation
- Saturated residual addition
- Output prediction
- Threshold-based anomaly detection

### SRAM-Backed Ring Buffers

Feature history is stored in synchronous **512 × 32 SKY130 SRAM macros**. Four INT8 channels are packed into each 32-bit SRAM word, allowing all allocated memory capacity to be used for signal history.

The ring-buffer architecture provides constant-time access to delayed samples required by each dilation level while avoiding large register arrays and data movement through software-managed buffers.

### AXI4-Stream Interface

The top-level wrapper uses **AXI4-Stream** semantics:

- `s_tvalid` / `s_tready` input handshake
- `m_tvalid` / `m_tready` output handshake
- Backpressure-aware data movement
- Cycle-aligned valid propagation
- Streaming sample-by-sample inference

This interface allows the accelerator to be integrated into a larger FPGA, ASIC, or SoC datapath without redesigning the neural-network core.

---

## Depthwise vs. Channel-Mixing Study

Two residual-block architectures were implemented and synthesized to quantify the hardware cost of cross-channel feature extraction.

| Metric | Depthwise Block | Channel-Mixing Block |
|---|---:|---:|
| Maximum clock frequency | **200 MHz** | **172 MHz** |
| Latency | **9 cycles** | **99 cycles** |
| Effective throughput | **6.4 GMAC/s** | **2.67 GMAC/s** |
| Logic area | **0.208 mm²** | **0.269 mm²** |
| Estimated energy per block | **5.8 nJ** | **62.86 nJ** |

Relative to channel mixing, the depthwise architecture achieved:

- **11× lower latency**
- **16× fewer MAC operations**
- **22.7% lower logic area**
- **10.8× lower estimated energy**

Channel mixing improved mean prediction MSE by **11.9% to 15.5%** across the evaluated motion environments, establishing a clear accuracy-versus-hardware-cost tradeoff.

---

## Model Training and Software Reference

The model was trained in **Python with PyTorch** using the Oxford Inertial Odometry Dataset (**OxIOD**). The training flow supports multichannel window generation, normalization, paired-seed evaluation, and architecture comparison.

A streaming **C++ inference engine** was implemented before the RTL to validate causal buffering, dilation indexing, residual behavior, and sample-by-sample execution. The floating-point C++ implementation matched the PyTorch reference to approximately **1.77 × 10⁻⁷ RMSE**.

A separate bit-accurate fixed-point C++ golden model reproduces the hardware operations used for RTL verification, including:

- Signed INT8 multiplication
- 32-bit accumulation
- Weight ordering and causal tap evaluation
- Requantization and clipping
- ReLU
- Saturated residual addition
- Ring-buffer updates

---

## Verification

The RTL was verified in **Siemens QuestaSim** using modular and end-to-end self-checking SystemVerilog testbenches.

The verification environment includes:

- Directed corner-case testing
- Constrained-random input generation
- C++ golden-model comparisons
- Self-checking scoreboards
- Protocol and latency assertions
- Backpressure testing for AXI4-Stream
- Independent module-level testbenches
- End-to-end accelerator validation

More than **3,000 randomized scenarios** were executed across the datapath and streaming interface.

Verified components include:

- MAC arrays
- Quantization and ReLU units
- SRAM-backed ring buffers
- Depthwise residual blocks
- Channel-mixing residual blocks
- Input and output blocks
- Anomaly detector
- AXI4-Stream wrapper
- Full integrated pipeline

---

## ASIC Synthesis and RTL Quality

### Synopsys VC SpyGlass

RTL lint and static analysis were completed with **zero fatal issues and zero errors**. The checks were used to identify width mismatches, unintended truncation, incomplete assignments, and other synthesizability or maintainability risks before synthesis.

### Synopsys Design Compiler Ultra

The accelerator and residual-block variants were synthesized using **Synopsys Design Compiler Ultra** with the **SkyWater SKY130** standard-cell library and SRAM macros.

The synthesis flow was used to:

- Close timing at the target frequencies
- Identify critical combinational and SRAM-to-MAC paths
- Compare architecture latency, area, power, and throughput
- Estimate energy per residual-block evaluation
- Guide pipeline and datapath optimization
- Generate synthesized netlists and timing reports

The final integrated RTL closed synthesis with **zero timing violations**.

---

## Repository Structure

```text
docs/                  Architecture notes, implementation plans, and reports
results/
  logs/                Simulation, lint, and synthesis logs
  media/               Training and architecture-validation plots
  netlist/             Synopsys Design Compiler synthesized netlists
  reports/             Timing, area, power, and synthesis reports
src/
  cpp/                 Streaming and fixed-point C++ reference models
  python/              PyTorch training, validation, and evaluation scripts
  rtl/
    archive/            Retired SystemVerilog modules
    common/             Active SystemVerilog RTL modules
  scripts/
    powershell/         Windows automation scripts
    shell/              Linux/Unix build and simulation scripts
    tcl/                QuestaSim, SpyGlass, and Design Compiler automation
  tb/                   SystemVerilog testbenches and verification components
```

---

## Primary Tools

- **SystemVerilog** — RTL implementation and verification
- **Siemens QuestaSim** — simulation and randomized functional verification
- **Synopsys VC SpyGlass** — lint and static RTL analysis
- **Synopsys Design Compiler Ultra** — synthesis, timing closure, and PPA analysis
- **SkyWater SKY130 PDK** — standard cells and synchronous SRAM macros
- **C++** — streaming inference and bit-accurate golden modeling
- **Python / PyTorch** — model training and architecture evaluation
- **TCL, Bash, and PowerShell** — EDA and workflow automation
- **Git / GitHub** — version control and project management

---

## Research Basis and Dataset

The TCN architecture is based on:

- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271)

The architecture was trained and evaluated using the Oxford Inertial Odometry Dataset:

- [Oxford Inertial Odometry Dataset (OxIOD)](https://drive.google.com/file/d/1UCHY3ENCybcBNyiC2wx1gQEWSLqzJag0/view)

The depthwise-versus-channel-mixing architecture study is documented in:

- **R. Gaffere, “Depthwise vs. Channel-Mixing Residual Blocks: Hardware and Accuracy Evaluation,” IEEE HPEC 2026, under review.**

---

## Project Outcome

This project produced a complete, verified TCN accelerator rather than an isolated RTL kernel. It connects machine-learning model development, fixed-point numerical design, memory architecture, streaming interfaces, verification, and ASIC synthesis in one reproducible hardware/software workflow.

The final result demonstrates practical experience in **SystemVerilog RTL, neural-network acceleration, AXI4-Stream integration, SRAM macro design, fixed-point arithmetic, randomized verification, Synopsys VC SpyGlass, Synopsys Design Compiler, timing closure, and PPA-driven architecture optimization**.

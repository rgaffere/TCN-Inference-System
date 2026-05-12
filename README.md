# Real-Time Predictive Signal Hardware

Hardware/software co-design project implementing a Temporal Convolutional Network (TCN) for real-time multichannel signal prediction and anomaly detection.

The project includes:
- PyTorch model training
- C++ streaming inference engine
- SystemVerilog RTL implementation
- RTL simulation and verification flow
- ASIC-oriented design exploration using Synopsys tools

Based on the paper:  
https://arxiv.org/pdf/1803.01271

Dataset used for validation of architecture (OxIOD):  
https://drive.google.com/file/d/1UCHY3ENCybcBNyiC2wx1gQEWSLqzJag0/view

---

## Project Stages

### Stage 1 — Learn the Architecture (Done)
Studied the TCN architecture, residual blocks, dilated causal convolutions, and receptive field behavior.

### Stage 2 — C++ Inference Engine (Done)
Implemented a streaming inference engine in C++ to understand and validate the architecture before hardware implementation.

### Stage 3 — PyTorch Training (Done)
Trained a TCN model on multichannel IMU data using PyTorch.

### Stage 4 — Validation & Verification (Done)
Validated the C++ inference engine against PyTorch outputs and documented the results.

### Stage 5 — Hardware Implementation (In Progress)
Current phase:
- SystemVerilog RTL development
- Module-level verification
- Simulation using QuestaSim and VCS
- ASIC-oriented synthesis exploration using Synopsys Design Compiler

---

## Repository Structure

```text
docs/       -> reports, architecture notes, implementation plans
results/    -> plots, validation output, simulation logs
src/
  cpp/      -> C++ inference engine
  python/   -> PyTorch training and validation
  rtl/      -> SystemVerilog RTL modules
  scripts/  -> simulation and utility scripts
  tb/       -> SystemVerilog testbenches
```

## Tools & Technologies

- SystemVerilog
- Synopsys Design Compiler
- QuestaSim
- Synopsys VCS
- Synopsys Verdi
- VS Code
- Git/GitHub
- C++
- Python
- PyTorch

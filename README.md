# Cerberus: Explainable ML-Guided GPU Offload Profitability Predictor

[![SegFault 2026 Track P04](https://img.shields.io/badge/SegFault%202026-Track%20P04-blue)](https://github.com/wizardwithcodehazard/Cerberus)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenMP 4.5+](https://img.shields.io/badge/OpenMP-4.5%2B%20Target-green.svg)](https://www.openmp.org/)

> **"The intelligent gatekeeper guarding the GPU offload boundary — analyze C/C++ loops, predict profitability with Neuro-Symbolic ML, explain decisions with TreeSHAP, and gate OpenMP 4.5+ target directives automatically."**

Built for **SegFault 2026 (Track P04)** — *Innovations in Compiler Technology (IICT, IISc Bengaluru)*.

---

## Why Cerberus?

Offloading computation to the GPU is not always beneficial. Automatic or naive GPU parallelization frequently produces slower code than a sequential CPU baseline due to **interconnect transfer latency**, **insufficient data parallelism**, **SIMD memory non-coalescing**, and **kernel launch overhead**.

Like the three-headed guardian of myth, **Cerberus** guards the GPU boundary by inspecting three pillars before allowing an offload:
1. **Head 1 — Parallel Safety:** Detects loop-carried RAW/WAW dependencies.
2. **Head 2 — Hardware Physics:** Calculates Williams Roofline bounds & interconnect transfer penalties.
3. **Head 3 — Machine Learning:** Predicts speedup using an XGBoost cost model with TreeSHAP explanations.

```
                  C/C++ Source Code
                          │
                          ▼
             ┌─────────────────────────┐
             │ 1. AST Safety Analyzer  │ (RAW/WAW Hazard Detection)
             │   • Dependency checks   │
             │   • Cache reuse ratios  │
             └────────────┬────────────┘
                          │
                   Parallel Safe?
                    ┌─────┴─────┐
                   NO          YES
                    │           │
                    ▼           ▼
             ┌───────────┐ ┌─────────────────────────┐
             │ Keep CPU  │ │ 2. Neuro-Symbolic Model │ (Williams Roofline + XGBoost)
             │ Sequential│ │   • Transfer cost vs    │
             │           │ │     Compute throughput  │
             └───────────┘ └────────────┬────────────┘
                                        │
                             Profitable on Target GPU?
                                  ┌─────┴─────┐
                                 NO          YES
                                  │           │
                                  ▼           ▼
                           ┌───────────┐ ┌────────────────────────────────┐
                           │ Keep CPU  │ │ 3. OpenMP Pragma Injector    │
                           │ Sequential│ │    #pragma omp target teams    │
                           │           │ │    distribute parallel for     │
                           └───────────┘ └────────────────────────────────┘
```

---

## Key Features

* **AST Loop & Dependency Analysis:** Statically checks loops for loop-carried data hazards, calculating memory footprint, spatial/temporal data reuse ratios, SIMD coalescing scores, and arithmetic intensity (FLOPs / Byte).
* **Neuro-Symbolic Cost Modeling:** Combines classical Williams Roofline theoretical bounds with a gradient-boosted XGBoost regressor to predict wall-clock speedup (T_CPU / T_GPU).
* **TreeSHAP Feature Attribution:** Converts mathematical penalties (PCIe bus bottleneck, warp divergence, launch latency) into human-readable compiler explanations.
* **Cross-Architecture Hardware Sensitivity:** Distinguishes between Discrete GPUs over PCIe (e.g. NVIDIA RTX), Integrated GPUs on Unified Memory (e.g. AMD Radeon 680M / Intel Iris), and External GPUs over Thunderbolt.
* **OpenMP Target Offload Pragma Injection:** Automatically generates `#pragma omp target teams distribute parallel for` with `map(to:)`, `map(from:)`, and `reduction(...)` clauses.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/wizardwithcodehazard/Cerberus.git
cd Cerberus

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Linux/macOS: source venv/bin/activate

# Install dependencies & local package
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### 1. All-In-One Command (Analyze + Gate + Transform):
```bash
cerberus kernel.c -o kernel_opt.c
```

### 2. Deep Explainability Mode (--explain):
```bash
cerberus kernel.c -o kernel_opt.c --explain
```

### 3. Simulate Cross-Hardware Targets (--target):
```bash
# Simulate on Discrete NVIDIA RTX 4090
cerberus kernel.c -t dgpu_rtx4090

# Simulate on Integrated AMD / Intel iGPU (Zero-Copy)
cerberus kernel.c -t igpu_amd_radeon

# Simulate on External Thunderbolt eGPU
cerberus kernel.c -t egpu_thunderbolt
```

---

## Physical Hardware Benchmarking

Cerberus is trained on real physical GPU silicon using a zero-dependency native OpenCL engine:

```bash
# Run real physical hardware benchmark collection on your machine:
python -m scripts.collect_data

# Train the ML cost model with 5-Fold Cross-Validation:
python -m scripts.train_model
```

---

## Authors (Team seeplusplus)
* **Sahil Rane** ([@wizardwithcodehazard](https://github.com/wizardwithcodehazard))
* **Vedant Patil**
* **Gaurang**

---

## License
This project is licensed under the [MIT License](LICENSE).

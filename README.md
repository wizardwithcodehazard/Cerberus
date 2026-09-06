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

* **AST Loop & Dependency Analysis:** Statically checks loops for loop-carried data hazards, calculating memory footprint, spatial/temporal data reuse ratios, SIMD coalescing scores, and arithmetic intensity (FLOPs / Byte). Handles `#define` macros, `constexpr` constants, array precision scaling (`double`/`float`/`char`), domain tensor dimensions, and `while` loops.
* **Neuro-Symbolic Cost Modeling:** Combines classical Williams Roofline theoretical bounds with a gradient-boosted XGBoost regressor trained on 1,055 heterogeneous physical GPU runs with **95% Confidence Interval** uncertainty estimation.
* **TreeSHAP Feature Attribution:** Converts mathematical penalties (PCIe bus bottleneck, warp divergence, launch latency) into human-readable compiler explanations with local SHAP force contributions.
* **50+ GPU Silicon Database & Discovery:** Built-in hardware database (NVIDIA RTX 20/30/40/50, AMD RX 5000/6000/7000, Intel Arc/Iris) with automatic OpenCL + WMI micro-architectural discovery and live host CPU ISA detection (AVX2/AVX-512).
* **Multi-Target Pragma Generation (OpenMP & OpenACC):** Automatically generates `#pragma omp target teams distribute parallel for` or `#pragma acc parallel loop` with `map(to:)`, `copyin()`, `copyout()`, `reduction(...)`, and `[0:N]` array section bounds.
* **Smart Test Harness Filtering:** Automatically isolates production compute kernels from test validation suites (`validate_*`, `check_*`, `main()`), with `--include-tests` override.

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

## CLI Reference & Usage

### 1. Interactive Gating & Diagnostic Triage
```bash
# Launch interactive analysis on candidate source code:
python -m cerberus.cli benchmarks/synthetic/test1.cpp
```

### 2. Automated Non-Interactive Batch Mode (`--batch`)
```bash
# Scan, inject OpenMP offload pragmas, generate optimization report, and exit:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --batch -o test1_offloaded.cpp
```

### 3. OpenACC Directive Generation (`--format openacc`)
```bash
# Generate OpenACC 2.7+ parallel loop directives:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --format openacc --batch -o test1_acc.cpp
```

### 4. CI/CD Machine-Readable JSON Output (`--json`)
```bash
# Export full loop features, Roofline bounds, TreeSHAP values, and predictions as JSON:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --json --target dgpu_rtx3060
```

### 5. Cross-Target Hardware Simulation (`-t` / `--target`)
```bash
# List all preset hardware profiles:
python -m cerberus.cli --list-targets

# Simulate on Discrete NVIDIA RTX 4090:
python -m cerberus.cli benchmarks/synthetic/test1.cpp -t dgpu_rtx4090 --batch

# Simulate on Integrated AMD RDNA2 iGPU (Zero-Copy Shared Memory):
python -m cerberus.cli benchmarks/synthetic/test1.cpp -t igpu_amd_radeon --batch

# Simulate on External Thunderbolt eGPU:
python -m cerberus.cli benchmarks/synthetic/test1.cpp -t egpu_thunderbolt --batch
```

### 6. Parametric Workload Scaling & Crossover Curve (`--sweep`)
```bash
# Display CPU-to-GPU speedup crossover across problem sizes N=100 to 10M:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --sweep

# Custom sweep range:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --sweep --sweep-range 512 1048576
```

### 7. Deep Explainability & Single-Loop Audit (`--explain`, `--audit`)
```bash
# Print TreeSHAP attribution factor breakdown for all loops:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --explain

# Deep micro-architectural audit panel for a specific loop index:
python -m cerberus.cli benchmarks/synthetic/test1.cpp --audit 1
```

---

## Physical Hardware Benchmarking & Training

Cerberus is trained on physical GPU silicon runs using a native OpenCL engine:

```bash
# 1. Collect empirical micro-benchmarking data on local silicon:
python -m scripts.collect_data

# 2. Train the XGBoost cost model with 5-Fold Stratified Cross-Validation:
python -m scripts.train_model

# 3. Run full test suite:
pytest tests/ -v
```

---

## Authors (Team seeplusplus)
* **Sahil Rane** ([@wizardwithcodehazard](https://github.com/wizardwithcodehazard))
* **Vedant Patil**
* **Gaurang**

---

## License
This project is licensed under the [MIT License](LICENSE).

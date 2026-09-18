# Cerberus Compiler Optimization & GPU Profitability Report

**Source File:** `benchmarks/synthetic/test1.cpp`  
**Target Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (DGPU)  
**Host Processor:** Host Processor (~0.614 TFLOPS FP32 Peak)  
**Memory Bandwidth:** 15.75 GB/s (PCIe Interconnect)  
**Compute Capacity:** 6.03 TFLOPS  
**Target Dialect:** `OPENMP` (Offload pragmas)  
**Cost Model:** XGBoost Regressor + TreeSHAP (Trained on 2,318 silicon runs -- CV ROC-AUC: 0.967, R2: 0.836)

## Executive Summary
- **Total Loop Regions Analyzed:** 10
- **GPU Offload Injected (Profitable):** 8 regions
- **CPU Sequential Preserved (Slowdowns Prevented):** 2 regions
- **Unsafe Race Hazards Blocked:** 0 regions

## Loop Optimization Gating Table

| Region | Depth | Dynamic Iterations | Effective AI | Roofline Attainable | Predicted Speedup (95% CI) | Gating Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 `matrix_multiply()` (L13-20) | 3 | 100,000,000,000 | 0.67 FLOP/B | 3.1 GFLOPS | 11.81x [2.4x-58.5x] | **INJECT OFFLOAD** |
| #2 `parallel_reduction()` (L29-31) | 1 | 10,000 | 0.75 FLOP/B | 11.8 GFLOPS | 0.26x [0.1x-1.3x] | KEEP CPU |
| #3 `nbody_update()` (L40-53) | 2 | 100,000,000 | 1.38 FLOP/B | 18.7 GFLOPS | 152.74x [30.9x-755.8x] | **INJECT OFFLOAD** |
| #4 `stencil_2d_convolution()` (L62-72) | 4 | 50,176 | 1.25 FLOP/B | 9.9 GFLOPS | 62.06x [12.5x-307.1x] | **INJECT OFFLOAD** |
| #5 `fft()` (L80-99) | 3 | 200,000,000 | 6.75 FLOP/B | 68.0 GFLOPS | 842.00x [170.2x-4166.4x] | **INJECT OFFLOAD** |
| #6 `sparse_mv_multiply()` (L107-113) | 2 | 100,000,000 | 0.15 FLOP/B | 1.4 GFLOPS | 12.10x [2.4x-59.9x] | KEEP CPU |
| #7 `pso_update()` (L124-134) | 2 | 30,000 | 0.88 FLOP/B | 8.8 GFLOPS | 54.62x [11.0x-270.3x] | **INJECT OFFLOAD** |
| #8 `compute_intensity_mix()` (L143-150) | 2 | 1,000,000 | 7.62 FLOP/B | 120.1 GFLOPS | 361.71x [73.1x-1789.8x] | **INJECT OFFLOAD** |
| #9 `conv_layer_forward()` (L162-186) | 7 | 36,864 | 1.94 FLOP/B | 19.5 GFLOPS | 252.23x [51.0x-1248.1x] | **INJECT OFFLOAD** |
| #10 `matrix_transpose()` (L194-198) | 2 | 1,048,576 | 0.50 FLOP/B | 4.0 GFLOPS | 19.66x [4.0x-97.3x] | **INJECT OFFLOAD** |

## Detailed Loop-by-Loop Micro-Architectural Audits

### Loop #1: `matrix_multiply()` (Lines 13-20)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 11.81x [95% CI: 2.4x-58.5x] | Roofline Ceiling: 3.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+3.94 SHAP), Loop Nesting Depth (+2.07 SHAP).
- **Speedup Estimate (95% CI):** 11.81x (Range: 2.39x - 58.45x | Confidence: 78%)
- **Workload:** 100,000,000,000 iterations | 800000.00 MFLOPs (8 FLOP/iter)
- **Memory Working Set:** 1200000.00 MB (66666666666.7x estimated cache reuse)
- **SIMD Coalescing Score:** 50% (Stride regularity: 0.60)
- **Theoretical Roofline Ceiling:** 3.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.938 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.068 SHAP` : Loop Nesting Depth
  - `+1.328 SHAP` : Total Compute Workload
  - `-0.959 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.476 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.331 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #2: `parallel_reduction()` (Lines 29-31)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.26x slowdown [95% CI: 0.05x-1.31x] | Roofline Ceiling: 11.8 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-0.59 SHAP), Loop Nesting Depth (-0.45 SHAP).
- **Speedup Estimate (95% CI):** 0.26x (Range: 0.05x - 1.31x | Confidence: 66%)
- **Workload:** 10,000 iterations | 0.03 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 0.04 MB (3750.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 11.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.831 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.220 SHAP` : Log2 Scaled Total Compute Workload
  - `+0.167 SHAP` : Host-Device Interconnect Bandwidth
  - `-0.592 SHAP` : Total Compute Workload
  - `-0.446 SHAP` : Loop Nesting Depth
  - `-0.346 SHAP` : Loop Trip Count & Parallelism

---

### Loop #3: `nbody_update()` (Lines 40-53)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 152.74x [95% CI: 30.9x-755.8x] | Roofline Ceiling: 18.7 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+4.02 SHAP), Loop Nesting Depth (+2.77 SHAP).
- **Speedup Estimate (95% CI):** 152.74x (Range: 30.87x - 755.82x | Confidence: 88%)
- **Workload:** 100,000,000 iterations | 2200.00 MFLOPs (22 FLOP/iter)
- **Memory Working Set:** 1600.00 MB (91666666.7x estimated cache reuse)
- **SIMD Coalescing Score:** 93% (Stride regularity: 0.93)
- **Theoretical Roofline Ceiling:** 18.7 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+4.022 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.771 SHAP` : Loop Nesting Depth
  - `+1.308 SHAP` : Total Compute Workload
  - `-0.681 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.207 SHAP` : Data Movement & Interconnect Overhead
  - `-0.165 SHAP` : Control Flow Branching Divergence Risk

---

### Loop #4: `stencil_2d_convolution()` (Lines 62-72)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 62.06x [95% CI: 12.5x-307.1x] | Roofline Ceiling: 9.9 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+3.75 SHAP), Loop Nesting Depth (+2.99 SHAP).
- **Speedup Estimate (95% CI):** 62.06x (Range: 12.54x - 307.07x | Confidence: 86%)
- **Workload:** 50,176 iterations | 0.75 MFLOPs (15 FLOP/iter)
- **Memory Working Set:** 0.60 MB (47040.0x estimated cache reuse)
- **SIMD Coalescing Score:** 69% (Stride regularity: 0.73)
- **Theoretical Roofline Ceiling:** 9.9 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.745 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.989 SHAP` : Loop Nesting Depth
  - `+0.496 SHAP` : Target GPU Architecture Class
  - `-0.121 SHAP` : Data Movement & Interconnect Overhead
  - `-0.083 SHAP` : Loop Trip Count & Parallelism
  - `-0.044 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #5: `fft()` (Lines 80-99)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 842.00x [95% CI: 170.2x-4166.4x] | Roofline Ceiling: 68.0 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+3.62 SHAP), Loop Nesting Depth (+2.51 SHAP).
- **Speedup Estimate (95% CI):** 842.00x (Range: 170.16x - 4166.44x | Confidence: 91%)
- **Workload:** 200,000,000 iterations | 10800.00 MFLOPs (54 FLOP/iter)
- **Memory Working Set:** 1600.00 MB (270000000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 68.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.619 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.506 SHAP` : Loop Nesting Depth
  - `+1.859 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.528 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.108 SHAP` : Log2 Scaled Trip Count
  - `-0.024 SHAP` : Parallel Reduction Accumulator

---

### Loop #6: `sparse_mv_multiply()` (Lines 107-113)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 12.10x slowdown [95% CI: 2.45x-59.89x] | Roofline Ceiling: 1.4 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Log2 Scaled Total Compute Workload (-0.77 SHAP), Data Movement & Interconnect Overhead (-0.70 SHAP).
- **Speedup Estimate (95% CI):** 12.10x (Range: 2.45x - 59.89x | Confidence: 78%)
- **Workload:** 100,000,000 iterations | 300.00 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 2000.00 MB (12500000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 77% (Stride regularity: 0.77)
- **Theoretical Roofline Ceiling:** 1.4 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.958 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.194 SHAP` : Loop Nesting Depth
  - `+1.232 SHAP` : Total Compute Workload
  - `-0.769 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.697 SHAP` : Data Movement & Interconnect Overhead
  - `-0.444 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #7: `pso_update()` (Lines 124-134)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 54.62x [95% CI: 11.0x-270.3x] | Roofline Ceiling: 8.8 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+3.88 SHAP), Loop Nesting Depth (+2.87 SHAP).
- **Speedup Estimate (95% CI):** 54.62x (Range: 11.04x - 270.27x | Confidence: 85%)
- **Workload:** 30,000 iterations | 0.42 MFLOPs (14 FLOP/iter)
- **Memory Working Set:** 0.48 MB (13125.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 8.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.877 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.873 SHAP` : Loop Nesting Depth
  - `+0.606 SHAP` : Target GPU Architecture Class
  - `-0.310 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.191 SHAP` : Loop Trip Count & Parallelism
  - `-0.025 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #8: `compute_intensity_mix()` (Lines 143-150)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 361.71x [95% CI: 73.1x-1789.8x] | Roofline Ceiling: 120.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+3.68 SHAP), Loop Nesting Depth (+2.51 SHAP).
- **Speedup Estimate (95% CI):** 361.71x (Range: 73.10x - 1789.83x | Confidence: 89%)
- **Workload:** 1,000,000 iterations | 61.00 MFLOPs (61 FLOP/iter)
- **Memory Working Set:** 8.00 MB (7625000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 120.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.680 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.509 SHAP` : Loop Nesting Depth
  - `+2.271 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.584 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.392 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.107 SHAP` : Host-Device Memory Footprint Volume

---

### Loop #9: `conv_layer_forward()` (Lines 162-186)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 252.23x [95% CI: 51.0x-1248.1x] | Roofline Ceiling: 19.5 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+4.01 SHAP), Loop Nesting Depth (+3.19 SHAP).
- **Speedup Estimate (95% CI):** 252.23x (Range: 50.97x - 1248.12x | Confidence: 89%)
- **Workload:** 36,864 iterations | 1.14 MFLOPs (31 FLOP/iter)
- **Memory Working Set:** 0.59 MB (71424.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 19.5 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+4.008 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+3.189 SHAP` : Loop Nesting Depth
  - `+0.762 SHAP` : Compute FLOPs Per Iteration
  - `-0.094 SHAP` : Loop Trip Count & Parallelism

---

### Loop #10: `matrix_transpose()` (Lines 194-198)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 19.66x [95% CI: 4.0x-97.3x] | Roofline Ceiling: 4.0 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+3.94 SHAP), Loop Nesting Depth (+2.56 SHAP).
- **Speedup Estimate (95% CI):** 19.66x (Range: 3.97x - 97.28x | Confidence: 81%)
- **Workload:** 1,048,576 iterations | 4.19 MFLOPs (4 FLOP/iter)
- **Memory Working Set:** 8.39 MB (524288.0x estimated cache reuse)
- **SIMD Coalescing Score:** 68% (Stride regularity: 0.75)
- **Theoretical Roofline Ceiling:** 4.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.942 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.564 SHAP` : Loop Nesting Depth
  - `+0.485 SHAP` : Target GPU Architecture Class
  - `-0.552 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.302 SHAP` : Data Movement & Interconnect Overhead
  - `-0.132 SHAP` : Williams Roofline Theoretical Upper Bound

---

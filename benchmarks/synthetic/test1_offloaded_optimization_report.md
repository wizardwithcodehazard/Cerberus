# Cerberus Compiler Optimization & GPU Profitability Report

**Source File:** `benchmarks/synthetic/test1.cpp`  
**Target Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (DGPU)  
**Host Processor:** AMD Ryzen 5 8645HS w/ Radeon 760M Graphics (~0.413 TFLOPS FP32 Peak)  
**Memory Bandwidth:** 15.75 GB/s (PCIe Interconnect)  
**Compute Capacity:** 6.03 TFLOPS  
**Target Dialect:** `OPENMP` (Offload pragmas)  
**Cost Model:** XGBoost Regressor + TreeSHAP (Trained on 1,055 silicon runs -- CV ROC-AUC: 0.901, R2: 0.548)

## Executive Summary
- **Total Loop Regions Analyzed:** 10
- **GPU Offload Injected (Profitable):** 7 regions
- **CPU Sequential Preserved (Slowdowns Prevented):** 3 regions
- **Unsafe Race Hazards Blocked:** 0 regions

## Loop Optimization Gating Table

| Region | Depth | Dynamic Iterations | Arithmetic Intensity | Roofline Attainable | Predicted Speedup (95% CI) | Gating Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 `matrix_multiply()` (L13-20) | 3 | 100,000,000,000 | 0.67 FLOP/B | 3.1 GFLOPS | 5.03x [0.5x-46.4x] | **INJECT OFFLOAD** |
| #2 `parallel_reduction()` (L29-31) | 1 | 10,000 | 0.75 FLOP/B | 11.8 GFLOPS | 0.53x [0.1x-4.9x] | KEEP CPU |
| #3 `nbody_update()` (L40-53) | 2 | 100,000,000 | 1.38 FLOP/B | 18.7 GFLOPS | 1.98x [0.2x-18.3x] | **INJECT OFFLOAD** |
| #4 `stencil_2d_convolution()` (L62-72) | 4 | 50,176 | 1.25 FLOP/B | 9.9 GFLOPS | 9.85x [1.1x-90.8x] | **INJECT OFFLOAD** |
| #5 `fft()` (L80-99) | 3 | 200,000,000 | 6.75 FLOP/B | 68.0 GFLOPS | 3.44x [0.4x-31.7x] | **INJECT OFFLOAD** |
| #6 `sparse_mv_multiply()` (L107-113) | 2 | 100,000,000 | 0.15 FLOP/B | 1.4 GFLOPS | 3.02x [0.3x-27.8x] | KEEP CPU |
| #7 `pso_update()` (L124-134) | 2 | 30,000 | 0.88 FLOP/B | 8.8 GFLOPS | 7.97x [0.9x-73.5x] | **INJECT OFFLOAD** |
| #8 `compute_intensity_mix()` (L143-150) | 2 | 1,000,000 | 7.62 FLOP/B | 120.1 GFLOPS | 2.55x [0.3x-23.5x] | **INJECT OFFLOAD** |
| #9 `conv_layer_forward()` (L162-186) | 7 | 36,864 | 1.94 FLOP/B | 19.5 GFLOPS | 3.10x [0.3x-28.6x] | KEEP CPU |
| #10 `matrix_transpose()` (L194-198) | 2 | 1,048,576 | 0.50 FLOP/B | 4.0 GFLOPS | 6.18x [0.7x-57.0x] | **INJECT OFFLOAD** |

## Detailed Loop-by-Loop Micro-Architectural Audits

### Loop #1: `matrix_multiply()` (Lines 13-20)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 5.03x [95% CI: 0.5x-46.4x] | Roofline Ceiling: 3.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.59 SHAP), Temporal/Spatial Cache Data Reuse (+1.39 SHAP).
- **Speedup Estimate (95% CI):** 5.03x (Range: 0.55x - 46.36x | Confidence: 70%)
- **Workload:** 100,000,000,000 iterations | 800000.00 MFLOPs (8 FLOP/iter)
- **Memory Working Set:** 1200000.00 MB (66666666666.7x estimated cache reuse)
- **SIMD Coalescing Score:** 50% (Stride regularity: 0.60)
- **Theoretical Roofline Ceiling:** 3.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.587 SHAP` : Total Compute Workload
  - `+1.387 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.994 SHAP` : Loop Trip Count & Parallelism
  - `-0.742 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.372 SHAP` : Host-Device Memory Footprint Volume
  - `-0.219 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #2: `parallel_reduction()` (Lines 29-31)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.53x slowdown [95% CI: 0.06x-4.89x] | Roofline Ceiling: 11.8 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-1.40 SHAP), Loop Trip Count & Parallelism (-0.61 SHAP).
- **Speedup Estimate (95% CI):** 0.53x (Range: 0.06x - 4.89x | Confidence: 55%)
- **Workload:** 10,000 iterations | 0.03 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 0.04 MB (3750.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 11.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.537 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.618 SHAP` : Target GPU Architecture Class
  - `+0.267 SHAP` : Compute FLOPs Per Iteration
  - `-1.402 SHAP` : Total Compute Workload
  - `-0.607 SHAP` : Loop Trip Count & Parallelism
  - `-0.079 SHAP` : SIMD Memory Coalescing Efficiency

---

### Loop #3: `nbody_update()` (Lines 40-53)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 1.98x [95% CI: 0.2x-18.3x] | Roofline Ceiling: 18.7 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+1.72 SHAP), Total Compute Workload (+1.48 SHAP).
- **Speedup Estimate (95% CI):** 1.98x (Range: 0.22x - 18.29x | Confidence: 55%)
- **Workload:** 100,000,000 iterations | 2200.00 MFLOPs (22 FLOP/iter)
- **Memory Working Set:** 1600.00 MB (91666666.7x estimated cache reuse)
- **SIMD Coalescing Score:** 93% (Stride regularity: 0.93)
- **Theoretical Roofline Ceiling:** 18.7 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.721 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.480 SHAP` : Total Compute Workload
  - `+0.902 SHAP` : Loop Trip Count & Parallelism
  - `-0.811 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.548 SHAP` : Host-Device Memory Footprint Volume
  - `-0.436 SHAP` : Compute FLOPs Per Iteration

---

### Loop #4: `stencil_2d_convolution()` (Lines 62-72)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 9.85x [95% CI: 1.1x-90.8x] | Roofline Ceiling: 9.9 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+2.44 SHAP), Target GPU Architecture Class (+1.18 SHAP).
- **Speedup Estimate (95% CI):** 9.85x (Range: 1.07x - 90.84x | Confidence: 77%)
- **Workload:** 50,176 iterations | 0.75 MFLOPs (15 FLOP/iter)
- **Memory Working Set:** 0.60 MB (47040.0x estimated cache reuse)
- **SIMD Coalescing Score:** 69% (Stride regularity: 0.73)
- **Theoretical Roofline Ceiling:** 9.9 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.441 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.179 SHAP` : Target GPU Architecture Class
  - `+0.352 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.140 SHAP` : Parallel Reduction Accumulator
  - `-0.133 SHAP` : Loop Trip Count & Parallelism
  - `-0.033 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #5: `fft()` (Lines 80-99)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 3.44x [95% CI: 0.4x-31.7x] | Roofline Ceiling: 68.0 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+1.68 SHAP), Total Compute Workload (+1.48 SHAP).
- **Speedup Estimate (95% CI):** 3.44x (Range: 0.37x - 31.71x | Confidence: 64%)
- **Workload:** 200,000,000 iterations | 10800.00 MFLOPs (54 FLOP/iter)
- **Memory Working Set:** 1600.00 MB (270000000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 68.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.680 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.477 SHAP` : Total Compute Workload
  - `+0.974 SHAP` : Loop Trip Count & Parallelism
  - `-0.691 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.638 SHAP` : Compute FLOPs Per Iteration
  - `-0.446 SHAP` : Host-Device Memory Footprint Volume

---

### Loop #6: `sparse_mv_multiply()` (Lines 107-113)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 3.02x slowdown [95% CI: 0.33x-27.83x] | Roofline Ceiling: 1.4 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Log2 Scaled Total Compute Workload (-1.00 SHAP), Host-Device Memory Footprint Volume (-0.64 SHAP).
- **Speedup Estimate (95% CI):** 3.02x (Range: 0.33x - 27.83x | Confidence: 61%)
- **Workload:** 100,000,000 iterations | 300.00 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 2000.00 MB (12500000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 77% (Stride regularity: 0.77)
- **Theoretical Roofline Ceiling:** 1.4 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.581 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.572 SHAP` : Total Compute Workload
  - `+0.910 SHAP` : Loop Trip Count & Parallelism
  - `-1.001 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.636 SHAP` : Host-Device Memory Footprint Volume
  - `-0.439 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #7: `pso_update()` (Lines 124-134)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 7.97x [95% CI: 0.9x-73.5x] | Roofline Ceiling: 8.8 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+2.20 SHAP), Target GPU Architecture Class (+1.52 SHAP).
- **Speedup Estimate (95% CI):** 7.97x (Range: 0.86x - 73.51x | Confidence: 75%)
- **Workload:** 30,000 iterations | 0.42 MFLOPs (14 FLOP/iter)
- **Memory Working Set:** 0.48 MB (13125.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 8.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.198 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.520 SHAP` : Target GPU Architecture Class
  - `+0.199 SHAP` : Compute FLOPs Per Iteration
  - `-0.209 SHAP` : Loop Trip Count & Parallelism
  - `-0.051 SHAP` : Parallel Reduction Accumulator
  - `-0.021 SHAP` : SIMD Memory Coalescing Efficiency

---

### Loop #8: `compute_intensity_mix()` (Lines 143-150)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 2.55x [95% CI: 0.3x-23.5x] | Roofline Ceiling: 120.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.50 SHAP), Temporal/Spatial Cache Data Reuse (+1.35 SHAP).
- **Speedup Estimate (95% CI):** 2.55x (Range: 0.28x - 23.50x | Confidence: 57%)
- **Workload:** 1,000,000 iterations | 61.00 MFLOPs (61 FLOP/iter)
- **Memory Working Set:** 8.00 MB (7625000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 120.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.496 SHAP` : Total Compute Workload
  - `+1.346 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.771 SHAP` : Target GPU Architecture Class
  - `-0.669 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.623 SHAP` : Compute FLOPs Per Iteration
  - `-0.166 SHAP` : SIMD Memory Coalescing Efficiency

---

### Loop #9: `conv_layer_forward()` (Lines 162-186)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 3.10x slowdown [95% CI: 0.34x-28.58x] | Roofline Ceiling: 19.5 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Compute FLOPs Per Iteration (-0.49 SHAP), Loop Trip Count & Parallelism (-0.16 SHAP).
- **Speedup Estimate (95% CI):** 3.10x (Range: 0.34x - 28.58x | Confidence: 62%)
- **Workload:** 36,864 iterations | 1.14 MFLOPs (31 FLOP/iter)
- **Memory Working Set:** 0.59 MB (71424.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 19.5 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.814 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.815 SHAP` : Target GPU Architecture Class
  - `+0.445 SHAP` : Total Compute Workload
  - `-0.491 SHAP` : Compute FLOPs Per Iteration
  - `-0.160 SHAP` : Loop Trip Count & Parallelism
  - `-0.126 SHAP` : Parallel Reduction Accumulator

---

### Loop #10: `matrix_transpose()` (Lines 194-198)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 6.18x [95% CI: 0.7x-57.0x] | Roofline Ceiling: 4.0 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+1.58 SHAP), Target GPU Architecture Class (+1.23 SHAP).
- **Speedup Estimate (95% CI):** 6.18x (Range: 0.67x - 56.96x | Confidence: 72%)
- **Workload:** 1,048,576 iterations | 4.19 MFLOPs (4 FLOP/iter)
- **Memory Working Set:** 8.39 MB (524288.0x estimated cache reuse)
- **SIMD Coalescing Score:** 68% (Stride regularity: 0.75)
- **Theoretical Roofline Ceiling:** 4.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.583 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.226 SHAP` : Target GPU Architecture Class
  - `+0.720 SHAP` : Total Compute Workload
  - `-0.299 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.098 SHAP` : Williams Roofline Theoretical Upper Bound
  - `-0.066 SHAP` : Log2 Scaled Trip Count

---

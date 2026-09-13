# Cerberus Compiler Optimization & GPU Profitability Report

**Source File:** `benchmarks/synthetic/test1.cpp`  
**Target Hardware:** AMD Radeon(TM) 680M (IGPU)  
**Host Processor:** AMD Ryzen 7 7735HS with Radeon Graphics (~0.41 TFLOPS FP32 Peak)  
**Memory Bandwidth:** 102.4 GB/s (Shared System Memory)  
**Compute Capacity:** 3.38 TFLOPS  
**Target Dialect:** `OPENMP` (Offload pragmas)  
**Cost Model:** XGBoost Regressor + TreeSHAP (Trained on 1,055 silicon runs -- CV ROC-AUC: 0.901, R2: 0.543)

## Executive Summary
- **Total Loop Regions Analyzed:** 10
- **GPU Offload Injected (Profitable):** 5 regions
- **CPU Sequential Preserved (Slowdowns Prevented):** 5 regions
- **Unsafe Race Hazards Blocked:** 0 regions

## Loop Optimization Gating Table

| Region | Depth | Dynamic Iterations | Effective AI | Roofline Attainable | Predicted Speedup (95% CI) | Gating Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 `matrix_multiply()` (L13-20) | 3 | 100,000,000,000 | 0.67 FLOP/B | 20.5 GFLOPS | 4.21x [0.5x-39.3x] | **INJECT OFFLOAD** |
| #2 `parallel_reduction()` (L29-31) | 1 | 10,000 | 0.75 FLOP/B | 76.8 GFLOPS | 0.45x [0.0x-4.2x] | KEEP CPU |
| #3 `nbody_update()` (L40-53) | 2 | 100,000,000 | 1.38 FLOP/B | 121.8 GFLOPS | 1.99x [0.2x-18.6x] | **INJECT OFFLOAD** |
| #4 `stencil_2d_convolution()` (L62-72) | 4 | 50,176 | 1.25 FLOP/B | 64.5 GFLOPS | 1.70x [0.2x-15.9x] | KEEP CPU |
| #5 `fft()` (L80-99) | 3 | 200,000,000 | 6.75 FLOP/B | 442.4 GFLOPS | 4.40x [0.5x-41.1x] | **INJECT OFFLOAD** |
| #6 `sparse_mv_multiply()` (L107-113) | 2 | 100,000,000 | 0.15 FLOP/B | 9.1 GFLOPS | 1.24x [0.1x-11.6x] | **INJECT OFFLOAD** |
| #7 `pso_update()` (L124-134) | 2 | 30,000 | 0.88 FLOP/B | 57.3 GFLOPS | 0.52x [0.1x-4.8x] | KEEP CPU |
| #8 `compute_intensity_mix()` (L143-150) | 2 | 1,000,000 | 7.62 FLOP/B | 780.8 GFLOPS | 3.06x [0.3x-28.6x] | KEEP CPU |
| #9 `conv_layer_forward()` (L162-186) | 7 | 36,864 | 1.94 FLOP/B | 127.0 GFLOPS | 1.64x [0.2x-15.3x] | KEEP CPU |
| #10 `matrix_transpose()` (L194-198) | 2 | 1,048,576 | 0.50 FLOP/B | 26.1 GFLOPS | 2.83x [0.3x-26.4x] | **INJECT OFFLOAD** |

## Detailed Loop-by-Loop Micro-Architectural Audits

### Loop #1: `matrix_multiply()` (Lines 13-20)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 4.21x [95% CI: 0.5x-39.3x] | Roofline Ceiling: 20.5 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+2.34 SHAP), Loop Trip Count & Parallelism (+1.24 SHAP).
- **Speedup Estimate (95% CI):** 4.21x (Range: 0.45x - 39.32x | Confidence: 67%)
- **Workload:** 100,000,000,000 iterations | 800000.00 MFLOPs (8 FLOP/iter)
- **Memory Working Set:** 1200000.00 MB (66666666666.7x estimated cache reuse)
- **SIMD Coalescing Score:** 50% (Stride regularity: 0.60)
- **Theoretical Roofline Ceiling:** 20.5 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.337 SHAP` : Total Compute Workload
  - `+1.236 SHAP` : Loop Trip Count & Parallelism
  - `+1.143 SHAP` : Temporal/Spatial Cache Data Reuse
  - `-1.235 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.427 SHAP` : Target GPU Architecture Class
  - `-0.156 SHAP` : Log2 Scaled Trip Count

---

### Loop #2: `parallel_reduction()` (Lines 29-31)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.45x slowdown [95% CI: 0.05x-4.17x] | Roofline Ceiling: 76.8 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-1.64 SHAP), Loop Trip Count & Parallelism (-0.57 SHAP).
- **Speedup Estimate (95% CI):** 0.45x (Range: 0.05x - 4.17x | Confidence: 55%)
- **Workload:** 10,000 iterations | 0.03 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 0.04 MB (3750.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 76.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.631 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.513 SHAP` : Williams Roofline Theoretical Upper Bound
  - `+0.303 SHAP` : Log2 Scaled Total Compute Workload
  - `-1.644 SHAP` : Total Compute Workload
  - `-0.568 SHAP` : Loop Trip Count & Parallelism
  - `-0.411 SHAP` : Target GPU Architecture Class

---

### Loop #3: `nbody_update()` (Lines 40-53)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 1.99x [95% CI: 0.2x-18.6x] | Roofline Ceiling: 121.8 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+2.07 SHAP), Temporal/Spatial Cache Data Reuse (+1.33 SHAP).
- **Speedup Estimate (95% CI):** 1.99x (Range: 0.21x - 18.64x | Confidence: 55%)
- **Workload:** 100,000,000 iterations | 2200.00 MFLOPs (22 FLOP/iter)
- **Memory Working Set:** 1600.00 MB (91666666.7x estimated cache reuse)
- **SIMD Coalescing Score:** 93% (Stride regularity: 0.93)
- **Theoretical Roofline Ceiling:** 121.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.069 SHAP` : Total Compute Workload
  - `+1.327 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.016 SHAP` : Loop Trip Count & Parallelism
  - `-1.002 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.619 SHAP` : Control Flow Branching Divergence Risk
  - `-0.462 SHAP` : Target GPU Architecture Class

---

### Loop #4: `stencil_2d_convolution()` (Lines 62-72)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 1.70x slowdown [95% CI: 0.18x-15.92x] | Roofline Ceiling: 64.5 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Target GPU Architecture Class (-0.60 SHAP), Loop Trip Count & Parallelism (-0.41 SHAP).
- **Speedup Estimate (95% CI):** 1.70x (Range: 0.18x - 15.92x | Confidence: 55%)
- **Workload:** 50,176 iterations | 0.75 MFLOPs (15 FLOP/iter)
- **Memory Working Set:** 0.60 MB (47040.0x estimated cache reuse)
- **SIMD Coalescing Score:** 69% (Stride regularity: 0.73)
- **Theoretical Roofline Ceiling:** 64.5 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.274 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.540 SHAP` : Williams Roofline Theoretical Upper Bound
  - `+0.495 SHAP` : Loop Nesting Depth
  - `-0.595 SHAP` : Target GPU Architecture Class
  - `-0.413 SHAP` : Loop Trip Count & Parallelism
  - `-0.080 SHAP` : Host-Device Interconnect Bandwidth

---

### Loop #5: `fft()` (Lines 80-99)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 4.40x [95% CI: 0.5x-41.1x] | Roofline Ceiling: 442.4 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+2.04 SHAP), Loop Trip Count & Parallelism (+1.15 SHAP).
- **Speedup Estimate (95% CI):** 4.40x (Range: 0.47x - 41.10x | Confidence: 68%)
- **Workload:** 200,000,000 iterations | 10800.00 MFLOPs (54 FLOP/iter)
- **Memory Working Set:** 1600.00 MB (270000000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 442.4 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.038 SHAP` : Total Compute Workload
  - `+1.153 SHAP` : Loop Trip Count & Parallelism
  - `+1.122 SHAP` : Temporal/Spatial Cache Data Reuse
  - `-1.139 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.395 SHAP` : Compute FLOPs Per Iteration
  - `-0.282 SHAP` : Target GPU Architecture Class

---

### Loop #6: `sparse_mv_multiply()` (Lines 107-113)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 1.24x [95% CI: 0.1x-11.6x] | Roofline Ceiling: 9.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+2.34 SHAP), Temporal/Spatial Cache Data Reuse (+1.74 SHAP).
- **Speedup Estimate (95% CI):** 1.24x (Range: 0.13x - 11.63x | Confidence: 55%)
- **Workload:** 100,000,000 iterations | 300.00 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 2000.00 MB (12500000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 77% (Stride regularity: 0.77)
- **Theoretical Roofline Ceiling:** 9.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.345 SHAP` : Total Compute Workload
  - `+1.743 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.207 SHAP` : Loop Trip Count & Parallelism
  - `-1.552 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.628 SHAP` : Host-Device Memory Footprint Volume
  - `-0.597 SHAP` : Target GPU Architecture Class

---

### Loop #7: `pso_update()` (Lines 124-134)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.52x slowdown [95% CI: 0.06x-4.82x] | Roofline Ceiling: 57.3 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Target GPU Architecture Class (-0.85 SHAP), Loop Trip Count & Parallelism (-0.42 SHAP).
- **Speedup Estimate (95% CI):** 0.52x (Range: 0.06x - 4.82x | Confidence: 55%)
- **Workload:** 30,000 iterations | 0.42 MFLOPs (14 FLOP/iter)
- **Memory Working Set:** 0.48 MB (13125.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 57.3 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.544 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.518 SHAP` : Williams Roofline Theoretical Upper Bound
  - `+0.237 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.845 SHAP` : Target GPU Architecture Class
  - `-0.419 SHAP` : Loop Trip Count & Parallelism
  - `-0.190 SHAP` : Arithmetic Intensity (FLOP/Byte)

---

### Loop #8: `compute_intensity_mix()` (Lines 143-150)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 3.06x slowdown [95% CI: 0.33x-28.61x] | Roofline Ceiling: 780.8 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Log2 Scaled Total Compute Workload (-0.92 SHAP), Target GPU Architecture Class (-0.54 SHAP).
- **Speedup Estimate (95% CI):** 3.06x (Range: 0.33x - 28.61x | Confidence: 62%)
- **Workload:** 1,000,000 iterations | 61.00 MFLOPs (61 FLOP/iter)
- **Memory Working Set:** 8.00 MB (7625000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 780.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.260 SHAP` : Total Compute Workload
  - `+1.493 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.728 SHAP` : Williams Roofline Theoretical Upper Bound
  - `-0.920 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.536 SHAP` : Target GPU Architecture Class
  - `-0.411 SHAP` : Compute FLOPs Per Iteration

---

### Loop #9: `conv_layer_forward()` (Lines 162-186)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 1.64x slowdown [95% CI: 0.18x-15.32x] | Roofline Ceiling: 127.0 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Target GPU Architecture Class (-0.52 SHAP), Loop Trip Count & Parallelism (-0.43 SHAP).
- **Speedup Estimate (95% CI):** 1.64x (Range: 0.18x - 15.32x | Confidence: 55%)
- **Workload:** 36,864 iterations | 1.14 MFLOPs (31 FLOP/iter)
- **Memory Working Set:** 0.59 MB (71424.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.80)
- **Theoretical Roofline Ceiling:** 127.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.184 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.664 SHAP` : Williams Roofline Theoretical Upper Bound
  - `+0.534 SHAP` : Total Compute Workload
  - `-0.518 SHAP` : Target GPU Architecture Class
  - `-0.426 SHAP` : Loop Trip Count & Parallelism
  - `-0.329 SHAP` : Compute FLOPs Per Iteration

---

### Loop #10: `matrix_transpose()` (Lines 194-198)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 2.83x [95% CI: 0.3x-26.4x] | Roofline Ceiling: 26.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+1.66 SHAP), Total Compute Workload (+0.73 SHAP).
- **Speedup Estimate (95% CI):** 2.83x (Range: 0.30x - 26.41x | Confidence: 60%)
- **Workload:** 1,048,576 iterations | 4.19 MFLOPs (4 FLOP/iter)
- **Memory Working Set:** 8.39 MB (524288.0x estimated cache reuse)
- **SIMD Coalescing Score:** 68% (Stride regularity: 0.75)
- **Theoretical Roofline Ceiling:** 26.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.658 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.725 SHAP` : Total Compute Workload
  - `+0.719 SHAP` : Loop Trip Count & Parallelism
  - `-0.801 SHAP` : Target GPU Architecture Class
  - `-0.535 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.167 SHAP` : Loop Nesting Depth

---

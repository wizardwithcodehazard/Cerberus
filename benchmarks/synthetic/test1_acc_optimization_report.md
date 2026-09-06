# Cerberus Compiler Optimization & GPU Profitability Report

**Source File:** `benchmarks/synthetic/test1.cpp`  
**Target Hardware:** AMD Radeon(TM) 680M (IGPU)  
**Host Processor:** AMD Ryzen 7 7735HS with Radeon Graphics (~0.41 TFLOPS FP32 Peak)  
**Memory Bandwidth:** 102.4 GB/s (Shared System Memory)  
**Compute Capacity:** 3.38 TFLOPS  
**Target Dialect:** `OPENACC` (Offload pragmas)  
**Cost Model:** XGBoost Regressor + TreeSHAP (Trained on 1,055 silicon runs -- CV ROC-AUC: 0.855, R2: 0.562)

## Executive Summary
- **Total Loop Regions Analyzed:** 17
- **GPU Offload Injected (Profitable):** 8 regions
- **CPU Sequential Preserved (Slowdowns Prevented):** 9 regions
- **Unsafe Race Hazards Blocked:** 0 regions

## Loop Optimization Gating Table

| Region | Depth | Dynamic Iterations | Effective AI | Roofline Attainable | Predicted Speedup | Gating Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 `MULTIPLICATION()` (L13-20) | 3 | 1,000,000,000,000 | 6666.67 FLOP/B | 3380.0 GFLOPS | 8.57x (76%) | **INJECT OFFLOAD** |
| #2 `REDUCTION()` (L29-31) | 1 | 10,000 | 3.00 FLOP/B | 307.2 GFLOPS | 0.12x (75%) | KEEP CPU |
| #3 `SIMULATION()` (L40-53) | 2 | 100,000,000 | 7.75 FLOP/B | 634.9 GFLOPS | 1.03x (55%) | KEEP CPU |
| #4 `CONVOLUTION()` (L62-72) | 4 | 99,960,004 | 0.67 FLOP/B | 5.1 GFLOPS | 8.45x (75%) | **INJECT OFFLOAD** |
| #5 `TRANSFORM()` (L80-99) | 3 | 99,980,000 | 16.00 FLOP/B | 1311.0 GFLOPS | 9.98x (77%) | **INJECT OFFLOAD** |
| #6 `MULTIPLY()` (L107-113) | 2 | 100,000,000 | 0.67 FLOP/B | 1.0 GFLOPS | 1.53x (55%) | **INJECT OFFLOAD** |
| #7 `OPTIMIZATION()` (L124-134) | 2 | 100,000,000 | 5.00 FLOP/B | 409.6 GFLOPS | 2.80x (60%) | **INJECT OFFLOAD** |
| #8 `TEST()` (L143-150) | 2 | 100,000,000 | 20.50 FLOP/B | 1679.4 GFLOPS | 2.53x (57%) | **INJECT OFFLOAD** |
| #9 `LAYER()` (L162-186) | 7 | 100,000,000,000,000,000,000 | 500000000000.00 FLOP/B | 3380.0 GFLOPS | 11.70x (78%) | **INJECT OFFLOAD** |
| #10 `TRANSPOSE()` (L194-198) | 2 | 100,000,000 | 2.00 FLOP/B | 35.8 GFLOPS | 0.30x (63%) | KEEP CPU |
| #11 `validate_matrix_multiply()` (L207-212) | 2 | 100,000,000 | 6.50 FLOP/B | 266.2 GFLOPS | 1.20x (55%) | **INJECT OFFLOAD** |
| #12 `validate_matrix_multiply()` (L216-220) | 2 | 100,000,000 | 15.00 FLOP/B | 614.4 GFLOPS | 0.42x (56%) | KEEP CPU |
| #13 `validate_fft()` (L260-263) | 1 | 10,000 | 11.50 FLOP/B | 1177.6 GFLOPS | 0.15x (73%) | KEEP CPU |
| #14 `validate_sparse()` (L278-280) | 1 | 10,000 | 6.00 FLOP/B | 614.4 GFLOPS | 0.07x (79%) | KEEP CPU |
| #15 `validate_convolution()` (L309-310) | 2 | 225 | 0.22 FLOP/B | 22.5 GFLOPS | 0.05x (82%) | KEEP CPU |
| #16 `validate_convolution()` (L315-317) | 1 | 9 | 0.11 FLOP/B | 10.8 GFLOPS | 0.05x (81%) | KEEP CPU |
| #17 `validate_transpose()` (L328-332) | 2 | 100,000,000 | 8.00 FLOP/B | 327.7 GFLOPS | 0.34x (61%) | KEEP CPU |

## Detailed Loop-by-Loop Micro-Architectural Audits

### Loop #1: `MULTIPLICATION()` (Lines 13-20)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 8.57x | Roofline Ceiling: 3380.0 GFLOPS [Compute-Bound]). Offload gated primarily by Total Compute Workload (+1.11 SHAP), Temporal/Spatial Cache Data Reuse (+0.91 SHAP).
- **Workload:** 1,000,000,000,000 iterations | 2000000.00 MFLOPs (2 FLOP/iter)
- **Memory Working Set:** 300.00 MB (10000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 15% (Stride regularity: 0.50)
- **Theoretical Roofline Ceiling:** 3380.0 GFLOPS (Compute-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.113 SHAP` : Total Compute Workload
  - `+0.910 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.853 SHAP` : Loop Nesting Depth
  - `-0.162 SHAP` : Data Movement & Interconnect Overhead
  - `-0.108 SHAP` : Memory Access Stride Regularity
  - `-0.022 SHAP` : Host-Device Interconnect Bandwidth

---

### Loop #2: `REDUCTION()` (Lines 29-31)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.12x slowdown | Roofline Ceiling: 307.2 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-1.11 SHAP), Data Movement & Interconnect Overhead (-0.24 SHAP).
- **Workload:** 10,000 iterations | 0.03 MFLOPs (3 FLOP/iter)
- **Memory Working Set:** 0.01 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 307.2 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+0.296 SHAP` : Compute FLOPs Per Iteration
  - `+0.219 SHAP` : Williams Roofline Theoretical Upper Bound
  - `+0.163 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-1.111 SHAP` : Total Compute Workload
  - `-0.242 SHAP` : Data Movement & Interconnect Overhead
  - `-0.183 SHAP` : Target GPU Architecture Class

---

### Loop #3: `SIMULATION()` (Lines 40-53)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 1.03x slowdown | Roofline Ceiling: 634.9 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Control Flow Branching Divergence Risk (-0.39 SHAP), SIMD Memory Coalescing Efficiency (-0.25 SHAP).
- **Workload:** 100,000,000 iterations | 3100.00 MFLOPs (31 FLOP/iter)
- **Memory Working Set:** 400.00 MB (6.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 634.9 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.534 SHAP` : Total Compute Workload
  - `+0.646 SHAP` : Loop Trip Count & Parallelism
  - `+0.567 SHAP` : Memory Access Stride Regularity
  - `-0.390 SHAP` : Control Flow Branching Divergence Risk
  - `-0.251 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.219 SHAP` : Target GPU Architecture Class

---

### Loop #4: `CONVOLUTION()` (Lines 62-72)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 8.45x | Roofline Ceiling: 5.1 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Temporal/Spatial Cache Data Reuse (+1.12 SHAP), Total Compute Workload (+1.02 SHAP).
- **Workload:** 99,960,004 iterations | 199.92 MFLOPs (2 FLOP/iter)
- **Memory Working Set:** 299.88 MB (9998.0x estimated cache reuse)
- **SIMD Coalescing Score:** 15% (Stride regularity: 0.50)
- **Theoretical Roofline Ceiling:** 5.1 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.120 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+1.019 SHAP` : Total Compute Workload
  - `+0.958 SHAP` : Loop Nesting Depth
  - `-0.196 SHAP` : Data Movement & Interconnect Overhead
  - `-0.183 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.107 SHAP` : Memory Access Stride Regularity

---

### Loop #5: `TRANSFORM()` (Lines 80-99)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 9.98x | Roofline Ceiling: 1311.0 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.41 SHAP), Temporal/Spatial Cache Data Reuse (+0.94 SHAP).
- **Workload:** 99,980,000 iterations | 4799.04 MFLOPs (48 FLOP/iter)
- **Memory Working Set:** 299.88 MB (9998.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 1311.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.406 SHAP` : Total Compute Workload
  - `+0.942 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.847 SHAP` : Loop Nesting Depth
  - `-0.207 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.141 SHAP` : Compute FLOPs Per Iteration
  - `-0.115 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #6: `MULTIPLY()` (Lines 107-113)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 1.53x | Roofline Ceiling: 1.0 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.08 SHAP), Loop Trip Count & Parallelism (+0.75 SHAP).
- **Workload:** 100,000,000 iterations | 200.00 MFLOPs (2 FLOP/iter)
- **Memory Working Set:** 300.00 MB (3.0x estimated cache reuse)
- **SIMD Coalescing Score:** 15% (Stride regularity: 0.10)
- **Theoretical Roofline Ceiling:** 1.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.080 SHAP` : Total Compute Workload
  - `+0.746 SHAP` : Loop Trip Count & Parallelism
  - `+0.350 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.216 SHAP` : Data Movement & Interconnect Overhead
  - `-0.206 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.131 SHAP` : Loop Nesting Depth

---

### Loop #7: `OPTIMIZATION()` (Lines 124-134)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 2.80x | Roofline Ceiling: 409.6 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.55 SHAP), Loop Trip Count & Parallelism (+0.62 SHAP).
- **Workload:** 100,000,000 iterations | 2000.00 MFLOPs (20 FLOP/iter)
- **Memory Working Set:** 400.00 MB (8.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 409.6 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.548 SHAP` : Total Compute Workload
  - `+0.624 SHAP` : Loop Trip Count & Parallelism
  - `+0.362 SHAP` : Memory Access Stride Regularity
  - `-0.140 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.095 SHAP` : Loop Nesting Depth
  - `-0.085 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #8: `TEST()` (Lines 143-150)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 2.53x | Roofline Ceiling: 1679.4 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.55 SHAP), Loop Trip Count & Parallelism (+0.75 SHAP).
- **Workload:** 100,000,000 iterations | 4100.00 MFLOPs (41 FLOP/iter)
- **Memory Working Set:** 200.00 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 1679.4 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.547 SHAP` : Total Compute Workload
  - `+0.748 SHAP` : Loop Trip Count & Parallelism
  - `+0.491 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.210 SHAP` : Temporal/Spatial Cache Data Reuse
  - `-0.201 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.145 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #9: `LAYER()` (Lines 162-186)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 11.70x | Roofline Ceiling: 3380.0 GFLOPS [Compute-Bound]). Offload gated primarily by Total Compute Workload (+1.22 SHAP), Temporal/Spatial Cache Data Reuse (+1.00 SHAP).
- **Workload:** 100,000,000,000,000,000,000 iterations | 200000000000000.00 MFLOPs (2 FLOP/iter)
- **Memory Working Set:** 400.00 MB (10000.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 3380.0 GFLOPS (Compute-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.221 SHAP` : Total Compute Workload
  - `+0.995 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.865 SHAP` : Loop Nesting Depth
  - `-0.130 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.109 SHAP` : Data Movement & Interconnect Overhead
  - `-0.008 SHAP` : Host-Device Interconnect Bandwidth

---

### Loop #10: `TRANSPOSE()` (Lines 194-198)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.30x slowdown | Roofline Ceiling: 35.8 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Compute FLOPs Per Iteration (-0.58 SHAP), Temporal/Spatial Cache Data Reuse (-0.43 SHAP).
- **Workload:** 100,000,000 iterations | 400.00 MFLOPs (4 FLOP/iter)
- **Memory Working Set:** 200.00 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 35% (Stride regularity: 0.50)
- **Theoretical Roofline Ceiling:** 35.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+0.896 SHAP` : Total Compute Workload
  - `+0.506 SHAP` : Loop Trip Count & Parallelism
  - `+0.289 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.584 SHAP` : Compute FLOPs Per Iteration
  - `-0.431 SHAP` : Temporal/Spatial Cache Data Reuse
  - `-0.418 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #11: `validate_matrix_multiply()` (Lines 207-212)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** GPU OFFLOAD PROFITABLE (Predicted Speedup: 1.20x | Roofline Ceiling: 266.2 GFLOPS [Interconnect-Constrained]). Offload gated primarily by Total Compute Workload (+1.29 SHAP), Loop Trip Count & Parallelism (+0.65 SHAP).
- **Workload:** 100,000,000 iterations | 1300.00 MFLOPs (13 FLOP/iter)
- **Memory Working Set:** 200.00 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.50)
- **Theoretical Roofline Ceiling:** 266.2 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.293 SHAP` : Total Compute Workload
  - `+0.654 SHAP` : Loop Trip Count & Parallelism
  - `+0.284 SHAP` : Compute FLOPs Per Iteration
  - `-0.288 SHAP` : Temporal/Spatial Cache Data Reuse
  - `-0.229 SHAP` : Target GPU Architecture Class
  - `-0.178 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #12: `validate_matrix_multiply()` (Lines 216-220)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.42x slowdown | Roofline Ceiling: 614.4 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Control Flow Branching Divergence Risk (-0.64 SHAP), Target GPU Architecture Class (-0.34 SHAP).
- **Workload:** 100,000,000 iterations | 1500.00 MFLOPs (15 FLOP/iter)
- **Memory Working Set:** 100.00 MB (1.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.50)
- **Theoretical Roofline Ceiling:** 614.4 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.501 SHAP` : Total Compute Workload
  - `+0.440 SHAP` : Loop Trip Count & Parallelism
  - `+0.319 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `-0.637 SHAP` : Control Flow Branching Divergence Risk
  - `-0.343 SHAP` : Target GPU Architecture Class
  - `-0.307 SHAP` : SIMD Memory Coalescing Efficiency

---

### Loop #13: `validate_fft()` (Lines 260-263)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.15x slowdown | Roofline Ceiling: 1177.6 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Target GPU Architecture Class (-0.43 SHAP), Data Movement & Interconnect Overhead (-0.24 SHAP).
- **Workload:** 10,000 iterations | 0.23 MFLOPs (23 FLOP/iter)
- **Memory Working Set:** 0.02 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 1177.6 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+0.235 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `+0.128 SHAP` : Total Compute Workload
  - `+0.099 SHAP` : Williams Roofline Theoretical Upper Bound
  - `-0.433 SHAP` : Target GPU Architecture Class
  - `-0.238 SHAP` : Data Movement & Interconnect Overhead
  - `-0.175 SHAP` : Loop Trip Count & Parallelism

---

### Loop #14: `validate_sparse()` (Lines 278-280)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.07x slowdown | Roofline Ceiling: 614.4 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-0.84 SHAP), Target GPU Architecture Class (-0.32 SHAP).
- **Workload:** 10,000 iterations | 0.12 MFLOPs (12 FLOP/iter)
- **Memory Working Set:** 0.02 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 614.4 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+0.110 SHAP` : Williams Roofline Theoretical Upper Bound
  - `+0.090 SHAP` : Arithmetic Intensity (FLOP/Byte)
  - `+0.014 SHAP` : SIMD Memory Coalescing Efficiency
  - `-0.844 SHAP` : Total Compute Workload
  - `-0.320 SHAP` : Target GPU Architecture Class
  - `-0.247 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #15: `validate_convolution()` (Lines 309-310)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.05x slowdown | Roofline Ceiling: 22.5 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-1.26 SHAP), Loop Trip Count & Parallelism (-0.39 SHAP).
- **Workload:** 225 iterations | 0.00 MFLOPs (1 FLOP/iter)
- **Memory Working Set:** 0.00 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 22.5 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+0.129 SHAP` : Compute FLOPs Per Iteration
  - `+0.035 SHAP` : Loop Nesting Depth
  - `+0.008 SHAP` : Memory Access Stride Regularity
  - `-1.257 SHAP` : Total Compute Workload
  - `-0.390 SHAP` : Loop Trip Count & Parallelism
  - `-0.341 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #16: `validate_convolution()` (Lines 315-317)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.05x slowdown | Roofline Ceiling: 10.8 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-1.29 SHAP), Loop Trip Count & Parallelism (-0.38 SHAP).
- **Workload:** 9 iterations | 0.00 MFLOPs (12 FLOP/iter)
- **Memory Working Set:** 0.00 MB (1.0x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 10.8 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+0.062 SHAP` : Williams Roofline Theoretical Upper Bound
  - `-1.286 SHAP` : Total Compute Workload
  - `-0.375 SHAP` : Loop Trip Count & Parallelism
  - `-0.272 SHAP` : Data Movement & Interconnect Overhead

---

### Loop #17: `validate_transpose()` (Lines 328-332)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.34x slowdown | Roofline Ceiling: 327.7 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Control Flow Branching Divergence Risk (-0.58 SHAP), Target GPU Architecture Class (-0.40 SHAP).
- **Workload:** 100,000,000 iterations | 1600.00 MFLOPs (16 FLOP/iter)
- **Memory Working Set:** 200.00 MB (2.0x estimated cache reuse)
- **SIMD Coalescing Score:** 80% (Stride regularity: 0.50)
- **Theoretical Roofline Ceiling:** 327.7 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.442 SHAP` : Total Compute Workload
  - `+0.457 SHAP` : Loop Trip Count & Parallelism
  - `+0.122 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.582 SHAP` : Control Flow Branching Divergence Risk
  - `-0.395 SHAP` : Target GPU Architecture Class
  - `-0.285 SHAP` : SIMD Memory Coalescing Efficiency

---

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-01

### Added
- **LLVM libclang Semantic AST Parser:** Direct extraction of 12 physical loop features including trip count, FLOPs per iteration, working-set memory footprint, arithmetic intensity, memory coalescing scores, and loop-carried data dependency hazards.
- **Physics-Constrained Hurdle XGBoost Cost Model:** Two-stage classification and monotonic regression model predicting GPU offload profitability with Williams Roofline theoretical throughput bounding.
- **Explainability & Attribution Engine:** Local TreeSHAP game-theoretic Shapley feature attribution with positive/negative factor scoring.
- **Multi-Dialect Pragma Transformer:** Source-to-source code synthesis for OpenMP 4.5+ (`#pragma omp target teams distribute parallel for`) and OpenACC (`#pragma acc parallel loop`) with dynamic runtime crossover problem size guards (`if(N >= crossover)`).
- **Direct C-ABI OpenCL Probing:** Micro-architectural hardware discovery querying compute units, boost clocks, PCIe interconnect bandwidth, and unified memory topology across NVIDIA, AMD, Intel, and Apple platforms.
- **Rich Terminal User Interface & CI/CD Gating:** Interactive inspection menu, parametric crossover sweeps (`--sweep`), deep loop audits (`--audit`), and machine-readable JSON output (`--json`).
- **Comprehensive Benchmark Suite:** 15+ high-performance compute kernels spanning linear algebra, stencils, image processing, and HMM probabilistic workloads.
- **Offline AI Optimization Advisor:** Local SLM integration and AST rule-based code refactoring suggestions.
- **Automated CI/CD Workflow:** Multi-version Python (3.9, 3.11) test pipeline on GitHub Actions.
- **Documentation & Community Files:** `CONTRIBUTING.md`, `SECURITY.md`, `cerberus/MODEL_CARD.md`, `dataset/README.md`, and issue/PR templates.

"""Offline AI Loop Optimizer & Refactoring Advisor Module (Local SLM + Compiler AST Hybrid)."""

import os
import re
from typing import List, Dict, Any, Optional
from cerberus.parser import LoopFeature
from cerberus.model import PredictionResult
from cerberus.hardware import HardwareProfile

class AILoopAdvisor:
    """Provides targeted micro-architectural refactoring advice using a local Small Language Model (SLM)

    grounded by TreeSHAP attribution and AST static metrics.
    """

    _model = None
    _tokenizer = None
    _model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    @classmethod
    def _init_local_slm(cls):
        """Lazy loader for the local 0.5B coding SLM (runs locally on CPU / iGPU)."""
        if cls._model is not None:
            return cls._model, cls._tokenizer

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load lightweight 0.5B coder model
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            cls._model = AutoModelForCausalLM.from_pretrained(
                cls._model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            return cls._model, cls._tokenizer
        except Exception:
            # Fallback if transformers or weights cannot be loaded
            return None, None

    @classmethod
    def generate_advice(cls, loop: LoopFeature, pred: PredictionResult, hw: HardwareProfile, use_local_slm: bool = True) -> Dict[str, Any]:
        """Analyzes bottleneck factors and generates exact code refactoring suggestions."""
        advice_items = []

        if pred.is_profitable:
            advice_items.append("Kernel is already GPU-profitable. Ready for OpenMP target pragma injection.")
            return {
                "status": "PROFITABLE",
                "bottleneck": "None (Compute & Parallelism Sufficient)",
                "advice": advice_items,
                "refactoring_code": None
            }

        shap = pred.shap_values
        bottlenecks = []

        # 1. Memory Coalescing & Indirect Gather
        if loop.coalescing_efficiency < 0.5 or loop.stride_regularity < 0.5:
            bottlenecks.append("Non-coalesced / indirect memory access causes warp thread serialization.")
            advice_items.append(
                "Memory Access Bottleneck: Group accesses into contiguous row-major blocks or stage indirect lookups into GPU Shared Memory (L1 cache)."
            )

        # 2. Control-Flow Branch Divergence
        if loop.branch_divergence_count > 0 or shap.get("branch_divergence_count", 0) < -0.15:
            bottlenecks.append("Control-flow branch divergence serializes GPU SIMD lanes.")
            advice_items.append(
                "Branch Divergence Bottleneck: Replace data-dependent branching with branchless conditional selects (std::clamp, fmaxf, ternary operators)."
            )

        # 3. Small Trip Count & Launch Latency
        if loop.trip_count < 10000 or shap.get("trip_count", 0) < -0.2:
            overhead_type = "shared memory synchronization" if hw.unified_memory else "PCIe host-device data transfer"
            bottlenecks.append(f"Trip count ({loop.trip_count:,}) is too small to amortize {overhead_type}.")
            advice_items.append(
                f"Workload Scale Bottleneck: Fuse consecutive small loops into a single batch kernel, or collapse multi-dimensional loops using 'collapse(2)'."
            )

        # 4. Low Arithmetic Intensity
        if loop.arithmetic_intensity < 1.0:
            bottlenecks.append(f"Low Arithmetic Intensity ({loop.arithmetic_intensity:.2f} FLOP/Byte) makes kernel strictly memory-bandwidth bound.")
            advice_items.append(
                f"Low Arithmetic Intensity: Increase computational density per memory fetch via temporal tiling or register unrolling."
            )

        if not advice_items:
            advice_items.append(f"Kernel execution time on {hw.name} is slower than CPU AVX baseline due to memory overhead.")

        # Generate Refactored Code using Local SLM
        refactored_code = None
        primary_bottleneck = " & ".join(bottlenecks) if bottlenecks else "Low arithmetic intensity and transfer latency"

        if use_local_slm:
            refactored_code = cls._generate_slm_refactoring(loop, hw, primary_bottleneck)

        # Fallback to AST-grounded dynamic synthesis if SLM is unavailable
        if not refactored_code:
            refactored_code = cls._synthesize_ast_refactoring(loop, hw)

        return {
            "status": "UNPROFITABLE",
            "bottleneck": primary_bottleneck,
            "advice": advice_items,
            "refactoring_code": refactored_code
        }

    @classmethod
    def _generate_slm_refactoring(cls, loop: LoopFeature, hw: HardwareProfile, bottleneck_str: str) -> Optional[str]:
        """Prompts the local SLM to generate a customized C/C++ refactored kernel."""
        model, tokenizer = cls._init_local_slm()
        if model is None or tokenizer is None:
            return None

        prompt = (
            f"<|im_start|>system\n"
            f"You are an expert GPU and High Performance Computing (HPC) Compiler Engineer. "
            f"Given a C/C++ loop that is UNPROFITABLE on GPU ({hw.name}), rewrite the code to fix the specific hardware bottleneck. "
            f"Output ONLY the refactored C/C++ code inside ```cpp ... ``` with concise comments.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Original Loop Code:\n"
            f"```cpp\n{loop.raw_source.strip()}\n```\n\n"
            f"Compiler Diagnostics:\n"
            f"- Target Hardware: {hw.name} ({hw.device_type.upper()})\n"
            f"- Arithmetic Intensity: {loop.arithmetic_intensity:.2f} FLOP/Byte\n"
            f"- Identified Bottleneck: {bottleneck_str}\n\n"
            f"Rewrite this loop into an optimized, SIMD/GPU-friendly version that eliminates this bottleneck.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            import torch
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=220,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract code block
            match = re.search(r"```cpp(.*?)```", generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return generated_text.strip()
        except Exception:
            return None

    @classmethod
    def _synthesize_ast_refactoring(cls, loop: LoopFeature, hw: HardwareProfile) -> str:
        """Dynamic AST-aware fallback code synthesis using exact variable names."""
        var = loop.iterator_variable or "i"
        upper = str(loop.trip_count) if loop.trip_count else "N"
        arrays_r = list(loop.arrays_read)
        arrays_w = list(loop.arrays_written)

        if loop.branch_divergence_count > 0:
            return (
                f"// AST-Synthesized Refactoring: Branchless Conditional Select\n"
                f"// Eliminates SIMD warp divergence across threads\n"
                f"#pragma omp target teams distribute parallel for\n"
                f"for (int {var} = 0; {var} < {upper}; ++{var}) {{\n"
                f"    // Replace data-dependent branching with branchless mask/select\n"
                f"    float cond = ({arrays_r[0] if arrays_r else 'data'}[{var}] > 0.0f) ? 1.0f : 0.0f;\n"
                f"    {arrays_w[0] if arrays_w else 'out'}[{var}] = cond * ({arrays_r[0] if arrays_r else 'data'}[{var}] * 2.0f);\n"
                f"}}"
            )
        elif loop.coalescing_efficiency < 0.5:
            return (
                f"// AST-Synthesized Refactoring: Coalesced Shared-Memory Staging\n"
                f"// Eliminates irregular stride penalty by prefetching into L1 local cache\n"
                f"#pragma omp target teams distribute parallel for\n"
                f"for (int {var} = 0; {var} < {upper}; {var} += 32) {{\n"
                f"    float local_tile[32];\n"
                f"    for (int k = 0; k < 32 && ({var} + k) < {upper}; ++k) {{\n"
                f"        local_tile[k] = {arrays_r[0] if arrays_r else 'input'}[{var} + k];\n"
                f"    }}\n"
                f"    // Process local_tile in fast GPU registers/L1\n"
                f"}}"
            )
        else:
            return (
                f"// AST-Synthesized Refactoring: Loop Collapse & Fusion\n"
                f"#pragma omp target teams distribute parallel for collapse(2)\n"
                f"for (int {var} = 0; {var} < {upper}; ++{var}) {{\n"
                f"    // Kernel body\n"
                f"}}"
            )

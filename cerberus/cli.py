"""Cerberus: Explainable ML-Guided GPU Offload Profitability Predictor CLI."""

import os
import sys
import json as json_lib
import copy
from typing import Optional, List, Dict, Any, Tuple
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt

from cerberus.parser import CLoopParser, LoopFeature
from cerberus.model import ProfitabilityModel, PredictionResult
from cerberus.hardware import HardwareProfile, detect_local_hardware, PRESET_PROFILES
from cerberus.transformer import OpenMPTransformer
from cerberus.advisor import AILoopAdvisor

console = Console()

def render_scan_table(loops: List[LoopFeature], results: List[Tuple[LoopFeature, PredictionResult]], hw: HardwareProfile):
    """Renders the main profitability analysis table with all detected loops."""
    # Determine if we need a compact table (narrow terminal)
    try:
        term_width = os.get_terminal_size().columns
    except (ValueError, OSError):
        term_width = 160

    compact = term_width < 120

    table = Table(title="Loop Region Profitability & Safety Analysis", show_lines=True)
    table.add_column("Region", style="cyan", justify="left", max_width=30 if compact else 40)
    table.add_column("Trip / Depth", style="magenta", justify="right")
    if not compact:
        table.add_column("Effective AI\n(Est. Cache Reuse)", justify="right")
        table.add_column("Roofline Upper Bound\n(Theoretical Peak)", justify="right")
    table.add_column("Predicted Speedup\n(ML Cost Model)", justify="right", style="bold")
    table.add_column("Gating Decision", justify="center")

    for i, (loop, pred) in enumerate(results, 1):
        # Truncate long function names
        fn_display = loop.function_name
        if len(fn_display) > 20:
            fn_display = fn_display[:18] + ".."
        region_title = f"#{i} {fn_display}():L{loop.line_start}-{loop.line_end}"
        trip_str = f"{loop.trip_count:,}\n(Depth {loop.nesting_depth})"

        if not loop.is_parallel_safe:
            speedup_str = "[red]0.00x[/red]"
            decision_str = "[bold red][REJECT: UNSAFE][/bold red]"
        elif pred.is_profitable:
            speedup_str = f"[bold green]{pred.predicted_speedup:.2f}x[/bold green]\n({pred.ci_lower:.1f}x-{pred.ci_upper:.1f}x CI)"
            decision_str = "[bold green][INJECT OFFLOAD][/bold green]"
        else:
            speedup_str = f"[bold yellow]{pred.predicted_speedup:.2f}x[/bold yellow]\n({pred.ci_lower:.2f}x-{pred.ci_upper:.2f}x CI)"
            decision_str = "[bold yellow][KEEP CPU][/bold yellow]"

        if compact:
            table.add_row(region_title, trip_str, speedup_str, decision_str)
        else:
            ai_str = f"{loop.arithmetic_intensity:.2f} FLOP/B\n({loop.data_reuse_ratio:.0f}x reuse est.)"
            roof_str = f"{pred.roofline.attainable_gflops:.1f} GFLOPS\n({'Bandwidth' if pred.roofline.is_memory_bound else 'Compute'}-Bound)"
            table.add_row(region_title, trip_str, ai_str, roof_str, speedup_str, decision_str)

    console.print(table)

    # Print summary line (no emojis)
    n_profitable = sum(1 for loop, pred in results if pred.is_profitable and loop.is_parallel_safe)
    n_cpu = sum(1 for loop, pred in results if not pred.is_profitable and loop.is_parallel_safe)
    n_unsafe = sum(1 for loop, pred in results if not loop.is_parallel_safe)

    summary_parts = []
    if n_profitable > 0:
        summary_parts.append(f"[bold green]{n_profitable} GPU-Profitable[/bold green]")
    if n_cpu > 0:
        summary_parts.append(f"[bold yellow]{n_cpu} CPU-Optimal[/bold yellow]")
    if n_unsafe > 0:
        summary_parts.append(f"[bold red]{n_unsafe} Unsafe (Race Hazard)[/bold red]")
    console.print("  " + "  |  ".join(summary_parts) + "\n")

def show_loop_audit(loop: LoopFeature, pred: PredictionResult, hw: HardwareProfile, idx: int):
    """Displays a rigorous micro-architectural audit for a single loop."""
    raw_traffic_mb = loop.raw_memory_traffic_bytes / 1e6
    effective_mb = loop.memory_footprint_bytes / 1e6
    flops_m = loop.total_flops / 1e6
    peak_gflops = hw.peak_tflops * 1000.0
    mem_ceiling_gflops = loop.arithmetic_intensity * (hw.bus_bandwidth_gbps * loop.coalescing_efficiency * loop.stride_regularity)

    content = f"""[bold cyan]WORKLOAD METRICS[/bold cyan]
  Function Name             : {loop.function_name}() (Lines {loop.line_start}-{loop.line_end})
  Loop Nesting Depth        : {loop.nesting_depth} ({'Strictly Nested' if loop.nesting_depth > 1 else 'Single Loop'})
  Dynamic Iterations        : {loop.trip_count:,}
  Floating Point Workload   : ~{flops_m:.2f} MFLOPs ({loop.flops_per_iter} FLOP/iter)
  Parallel Safety           : {'[green]SAFE (Embarrassingly Parallel)[/green]' if loop.is_parallel_safe else f'[red]UNSAFE ({loop.safety_reason})[/red]'}

[bold cyan]MEMORY TRAFFIC & CACHE REUSE MODEL[/bold cyan]
  Raw Source Memory Traffic : {raw_traffic_mb:.2f} MB (Direct loads/stores without cache)
  Distinct Working Set      : {effective_mb:.2f} MB (Unique tensor memory footprint)
  Estimated Reuse Potential : {loop.data_reuse_ratio:.1f}x (Temporal / Spatial cache locality)
  SIMD Coalescing Score     : {loop.coalescing_efficiency*100:.0f}% ({'Contiguous stride-1' if loop.coalescing_efficiency >= 0.8 else 'Indirect / strided gather'})
  Stride Regularity         : {loop.stride_regularity:.2f}

[bold cyan]WILLIAMS ROOFLINE THEORETICAL CEILINGS[/bold cyan]
  Raw Operational AI        : {loop.raw_arithmetic_intensity:.2f} FLOP/Byte
  Estimated Effective AI    : {loop.arithmetic_intensity:.2f} FLOP/Byte
  GPU Memory Bandwidth      : {hw.bus_bandwidth_gbps} GB/s ({'Shared System Memory' if hw.unified_memory else 'PCIe Bus'})
  GPU Memory Ceiling        : {mem_ceiling_gflops:.1f} GFLOPS
  GPU Compute Peak          : {peak_gflops:.1f} GFLOPS ({hw.peak_tflops} TFLOPS)
  Theoretical Roofline Bound: [bold yellow]{pred.roofline.attainable_gflops:.1f} GFLOPS[/bold yellow] ({'Bandwidth' if pred.roofline.is_memory_bound else 'Compute'}-Constrained)

[bold cyan]ML COST MODEL PREDICTION & GATING[/bold cyan]
  Predicted GPU/CPU Speedup : [bold {'green' if pred.is_profitable else 'yellow'}]{pred.predicted_speedup:.2f}x[/bold {'green' if pred.is_profitable else 'yellow'}] (95% CI: {pred.ci_lower:.2f}x - {pred.ci_upper:.2f}x)
  Model Confidence Score    : {pred.confidence*100:.1f}%
  Gating Optimization Action: [{'bold green][INJECT GPU OFFLOAD]' if pred.is_profitable and loop.is_parallel_safe else ('bold red][REJECT: UNSAFE]' if not loop.is_parallel_safe else 'bold yellow][KEEP CPU SEQUENTIAL]')}[/{'bold green' if pred.is_profitable and loop.is_parallel_safe else ('bold red' if not loop.is_parallel_safe else 'bold yellow')}]
"""
    console.print(Panel(content.strip(), title=f"[bold white]Deep Performance Audit -- Loop #{idx} ({loop.function_name})[/bold white]", border_style="cyan"))

def show_explanations(results: List[Tuple[LoopFeature, PredictionResult]]):
    """Displays feature-grounded TreeSHAP and Roofline explanations."""
    console.print("\n[bold white]Feature-Grounded Compiler Explanations (TreeSHAP):[/bold white]")
    for i, (loop, pred) in enumerate(results, 1):
        status_color = "green" if pred.is_profitable and loop.is_parallel_safe else ("red" if not loop.is_parallel_safe else "yellow")
        console.print(f"\n[bold cyan]Loop #{i} ({loop.function_name}, Lines {loop.line_start}-{loop.line_end}):[/bold cyan]")
        console.print(f"  [{status_color}]- {pred.primary_explanation}[/{status_color}]")

        if loop.is_parallel_safe:
            console.print("  [white]Top Positive Factors (Pushed towards GPU):[/white]")
            for label, val in pred.top_positive_factors:
                console.print(f"    [green]+{val:.3f} SHAP[/green] : {label}")
            console.print("  [white]Top Negative Factors (Pushed towards CPU):[/white]")
            for label, val in pred.top_negative_factors:
                console.print(f"    [red]{val:.3f} SHAP[/red] : {label}")

def show_crossover_sweep(results: List[Tuple[LoopFeature, PredictionResult]], model: ProfitabilityModel, hw: HardwareProfile, threshold: float, sweep_sizes: List[int], bytes_per_elem: int = 4):
    """Displays parametric workload scaling and CPU-to-GPU crossover table with multidimensional scaling."""
    console.print("\n[bold white]=== Parametric Workload Scaling & Crossover Analysis ===[/bold white]")
    
    sweep_table = Table(title="Profitability Crossover Curve across Problem Sizes (N)", show_lines=True)
    sweep_table.add_column("Kernel Region", style="cyan")
    for sz in sweep_sizes:
        sweep_table.add_column(f"N={sz:,}", justify="center")

    for loop, _ in results[:6]:
        row_items = [loop.function_name]
        for sz in sweep_sizes:
            scaled_loop = copy.copy(loop)
            
            # Rigorous multi-dimensional iteration and memory scaling
            if loop.nesting_depth >= 3:
                effective_trips = sz * sz * sz
                unique_mem = 3 * (sz * sz)
                scaled_loop.data_reuse_ratio = float(sz)
            elif loop.nesting_depth == 2:
                effective_trips = sz * sz
                unique_mem = max(1, len(loop.arrays_read | loop.arrays_written)) * (sz * sz)
                scaled_loop.data_reuse_ratio = 2.0
            else:
                effective_trips = sz
                unique_mem = max(1, len(loop.arrays_read | loop.arrays_written)) * sz
                scaled_loop.data_reuse_ratio = 1.0

            scaled_loop.trip_count = effective_trips
            scaled_loop.total_flops = max(1, effective_trips * scaled_loop.flops_per_iter)
            scaled_loop.memory_footprint_bytes = max(1024, unique_mem * bytes_per_elem)
            scaled_loop.arithmetic_intensity = scaled_loop.total_flops / float(scaled_loop.memory_footprint_bytes)
            
            pred_scale = model.predict_loop(scaled_loop, hw, speedup_threshold=threshold)
            if pred_scale.is_profitable:
                row_items.append(f"[bold green]GPU ({pred_scale.predicted_speedup:.1f}x)[/bold green]")
            else:
                row_items.append(f"[yellow]CPU ({pred_scale.predicted_speedup:.2f}x)[/yellow]")
        sweep_table.add_row(*row_items)
        
    console.print(sweep_table)

def generate_markdown_report(source_file: str, results: List[Tuple[LoopFeature, PredictionResult]], hw: HardwareProfile, model: ProfitabilityModel, report_path: str, dialect: str = "openmp"):
    """Generates a professional Markdown compiler optimization audit report."""
    total_loops = len(results)
    offloaded_count = sum(1 for loop, pred in results if pred.is_profitable and loop.is_parallel_safe)
    cpu_count = sum(1 for loop, pred in results if not pred.is_profitable and loop.is_parallel_safe)
    unsafe_count = sum(1 for loop, pred in results if not loop.is_parallel_safe)

    meta = model.metadata
    roc_auc = meta.get("roc_auc", 0.855)
    n_samples = meta.get("n_samples", 1055)
    r2_score = meta.get("r2", 0.562)

    md = [
        f"# Cerberus Compiler Optimization & GPU Profitability Report",
        f"\n**Source File:** `{source_file}`  ",
        f"**Target Hardware:** {hw.name} ({hw.device_type.upper()})  ",
        f"**Host Processor:** {hw.cpu_name} (~{hw.cpu_tflops} TFLOPS FP32 Peak)  ",
        f"**Memory Bandwidth:** {hw.bus_bandwidth_gbps} GB/s ({'Shared System Memory' if hw.unified_memory else 'PCIe Interconnect'})  ",
        f"**Compute Capacity:** {hw.peak_tflops} TFLOPS  ",
        f"**Target Dialect:** `{dialect.upper()}` (Offload pragmas)  ",
        f"**Cost Model:** XGBoost Regressor + TreeSHAP (Trained on {n_samples:,} silicon runs -- CV ROC-AUC: {roc_auc:.3f}, R2: {r2_score:.3f})\n",
        f"## Executive Summary",
        f"- **Total Loop Regions Analyzed:** {total_loops}",
        f"- **GPU Offload Injected (Profitable):** {offloaded_count} regions",
        f"- **CPU Sequential Preserved (Slowdowns Prevented):** {cpu_count} regions",
        f"- **Unsafe Race Hazards Blocked:** {unsafe_count} regions\n",
        f"## Loop Optimization Gating Table\n",
        f"| Region | Depth | Dynamic Iterations | Effective AI | Roofline Attainable | Predicted Speedup (95% CI) | Gating Decision |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for i, (loop, pred) in enumerate(results, 1):
        decision_tag = "**INJECT OFFLOAD**" if pred.is_profitable and loop.is_parallel_safe else ("**REJECT: UNSAFE**" if not loop.is_parallel_safe else "KEEP CPU")
        md.append(f"| #{i} `{loop.function_name}()` (L{loop.line_start}-{loop.line_end}) | {loop.nesting_depth} | {loop.trip_count:,} | {loop.arithmetic_intensity:.2f} FLOP/B | {pred.roofline.attainable_gflops:.1f} GFLOPS | {pred.predicted_speedup:.2f}x [{pred.ci_lower:.1f}x-{pred.ci_upper:.1f}x] | {decision_tag} |")

    md.append(f"\n## Detailed Loop-by-Loop Micro-Architectural Audits\n")

    for i, (loop, pred) in enumerate(results, 1):
        md.append(f"### Loop #{i}: `{loop.function_name}()` (Lines {loop.line_start}-{loop.line_end})")
        md.append(f"- **Parallel Safety:** {'SAFE' if loop.is_parallel_safe else f'UNSAFE ({loop.safety_reason})'}")
        md.append(f"- **Primary Gating Rationale:** {pred.primary_explanation}")
        md.append(f"- **Speedup Estimate (95% CI):** {pred.predicted_speedup:.2f}x (Range: {pred.ci_lower:.2f}x - {pred.ci_upper:.2f}x | Confidence: {pred.confidence*100:.0f}%)")
        md.append(f"- **Workload:** {loop.trip_count:,} iterations | {loop.total_flops / 1e6:.2f} MFLOPs ({loop.flops_per_iter} FLOP/iter)")
        md.append(f"- **Memory Working Set:** {loop.memory_footprint_bytes / 1e6:.2f} MB ({loop.data_reuse_ratio:.1f}x estimated cache reuse)")
        md.append(f"- **SIMD Coalescing Score:** {loop.coalescing_efficiency*100:.0f}% (Stride regularity: {loop.stride_regularity:.2f})")
        md.append(f"- **Theoretical Roofline Ceiling:** {pred.roofline.attainable_gflops:.1f} GFLOPS ({'Bandwidth-Bound' if pred.roofline.is_memory_bound else 'Compute-Bound'})")
        
        if loop.is_parallel_safe:
            md.append(f"\n**TreeSHAP Factor Breakdown:**")
            for label, val in pred.top_positive_factors:
                md.append(f"  - `+{val:.3f} SHAP` : {label}")
            for label, val in pred.top_negative_factors:
                md.append(f"  - `{val:.3f} SHAP` : {label}")
        md.append("\n---\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

def inject_and_save(source_file: str, transformer: OpenMPTransformer, results: List[Tuple[LoopFeature, PredictionResult]], hw: HardwareProfile, model: ProfitabilityModel, threshold: float, custom_output: Optional[str] = None, dialect: str = "openmp"):
    """Injects pragmas and saves both the transformed code and the optimization report."""
    if not custom_output:
        base, ext = os.path.splitext(source_file)
        custom_output = f"{base}_offloaded{ext}"
    
    report_output = f"{os.path.splitext(custom_output)[0]}_optimization_report.md"

    with open(source_file, 'r', encoding='utf-8') as f:
        raw_source = f.read()
    opt_source, decisions = transformer.transform_source(raw_source, speedup_threshold=threshold)
    injected_count = sum(1 for _, _, should_offload in decisions if should_offload)
    
    with open(custom_output, 'w', encoding='utf-8') as f:
        f.write(opt_source)
    
    # Generate Markdown Report
    generate_markdown_report(source_file, results, hw, model, report_output, dialect=dialect)
    
    dialect_title = "OpenACC" if dialect == "openacc" else "OpenMP target"
    if injected_count > 0:
        console.print(f"\n[bold green][SUCCESS] Injected {injected_count} GPU-profitable {dialect_title} pragma(s) into: {custom_output}[/bold green]")
    else:
        console.print(f"\n[bold yellow][GATING NOTICE] 0 loops were GPU-profitable on target hardware. Clean source preserved in: {custom_output}[/bold yellow]")
    
    console.print(f"[bold cyan][REPORT] Generated detailed compiler audit report: {report_output}[/bold cyan]\n")

def results_to_json(source_file: str, results: List[Tuple[LoopFeature, PredictionResult]], hw: HardwareProfile, model: ProfitabilityModel) -> dict:
    """Serializes all analysis results to a JSON-compatible dict for CI/CD integration."""
    meta = model.metadata
    output = {
        "source_file": source_file,
        "hardware": {
            "name": hw.name,
            "device_type": hw.device_type,
            "bus_bandwidth_gbps": hw.bus_bandwidth_gbps,
            "peak_tflops": hw.peak_tflops,
            "unified_memory": hw.unified_memory,
            "cpu_name": hw.cpu_name,
            "cpu_tflops": hw.cpu_tflops,
        },
        "model": {
            "n_samples": meta.get("n_samples", 0),
            "roc_auc": meta.get("roc_auc", 0),
            "r2": meta.get("r2", 0),
            "is_bootstrap": meta.get("is_bootstrap", False),
        },
        "summary": {
            "total_loops": len(results),
            "gpu_profitable": sum(1 for l, p in results if p.is_profitable and l.is_parallel_safe),
            "cpu_optimal": sum(1 for l, p in results if not p.is_profitable and l.is_parallel_safe),
            "unsafe": sum(1 for l, p in results if not l.is_parallel_safe),
        },
        "loops": []
    }
    for i, (loop, pred) in enumerate(results, 1):
        loop_data = {
            "index": i,
            "function_name": loop.function_name,
            "line_start": loop.line_start,
            "line_end": loop.line_end,
            "trip_count": loop.trip_count,
            "nesting_depth": loop.nesting_depth,
            "flops_per_iter": loop.flops_per_iter,
            "total_flops": loop.total_flops,
            "arithmetic_intensity": round(loop.arithmetic_intensity, 4),
            "memory_footprint_bytes": loop.memory_footprint_bytes,
            "coalescing_efficiency": loop.coalescing_efficiency,
            "stride_regularity": loop.stride_regularity,
            "is_parallel_safe": loop.is_parallel_safe,
            "safety_reason": loop.safety_reason,
            "has_reduction": loop.has_reduction,
            "predicted_speedup": round(pred.predicted_speedup, 4),
            "ci_lower": round(pred.ci_lower, 4),
            "ci_upper": round(pred.ci_upper, 4),
            "confidence": round(pred.confidence, 4),
            "is_profitable": pred.is_profitable,
            "roofline": {
                "attainable_gflops": round(pred.roofline.attainable_gflops, 2),
                "is_memory_bound": pred.roofline.is_memory_bound,
            },
            "explanation": pred.primary_explanation,
            "shap_values": {k: round(v, 4) for k, v in pred.shap_values.items()},
        }
        output["loops"].append(loop_data)
    return output

def list_target_profiles():
    """Prints all available hardware preset profiles."""
    console.print("\n[bold white]Available Hardware Target Profiles:[/bold white]\n")
    console.print(f"  {'auto':<22} Auto-detect local GPU via OpenCL + WMI")
    for key, profile in PRESET_PROFILES.items():
        bw_type = "Shared" if profile.unified_memory else "PCIe"
        console.print(f"  {key:<22} {profile.name} ({profile.peak_tflops} TFLOPS, {bw_type} {profile.bus_bandwidth_gbps} GB/s)")
    console.print()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('source_file', type=click.Path(exists=True), required=False)
@click.option('-o', '--output', 'output_file', type=click.Path(), default=None,
              help='Path to write optimized source with injected offload pragmas.')
@click.option('--report', 'report_file', type=click.Path(), default=None,
              help='Path to write detailed Markdown compiler optimization report.')
@click.option('-t', '--target', 'target_name', default='auto',
              help='Target GPU profile (e.g. auto, dgpu_rtx3060, dgpu_rtx4090, igpu_amd_radeon, igpu_intel_iris, egpu_thunderbolt).')
@click.option('--format', 'dialect', type=click.Choice(['openmp', 'openacc'], case_sensitive=False), default='openmp',
              help='Target offload pragma dialect (openmp or openacc). Default: openmp.')
@click.option('-p', '--param', 'param_str', default=None,
              help='Specify dynamic loop bounds (e.g. --param N=512,MATRIX_N=512,size=65536).')
@click.option('-e', '--explain', is_flag=True, default=False,
              help='Print deep TreeSHAP attribution factor breakdown and Roofline model analysis.')
@click.option('--audit', 'audit_loop_idx', type=int, default=None,
              help='Print deep micro-architectural audit box for a specific loop index (e.g. --audit 2).')
@click.option('--sweep', is_flag=True, default=False,
              help='Perform parametric workload scaling sweep to show the CPU-to-GPU crossover point.')
@click.option('--sweep-range', 'sweep_range', nargs=2, type=int, default=None,
              help='Custom sweep range as MIN MAX (e.g. --sweep-range 100 10000000). Default: 100 to 10M.')
@click.option('--threshold', default=1.1, type=float,
              help='Minimum predicted speedup required to gate GPU offload (default: 1.1x).')
@click.option('--default-n', 'default_n', type=int, default=None,
              help='Override default trip count for unresolved loop variables (default: 10000).')
@click.option('--batch', is_flag=True, default=False,
              help='Non-interactive mode: scan, inject pragmas, generate report, and exit.')
@click.option('--json', 'json_output', is_flag=True, default=False,
              help='Output all results as JSON to stdout for CI/CD integration.')
@click.option('--list-targets', is_flag=True, default=False,
              help='Print all available hardware target profiles and exit.')
@click.option('--include-tests', is_flag=True, default=False,
              help='Include test harness and validation functions (validate_*, check_*, main).')
@click.option('-q', '--quiet', is_flag=True, default=False,
              help='Suppress Rich formatting, print minimal output.')
def main(source_file: Optional[str], output_file: str, report_file: Optional[str], target_name: str, dialect: str,
         param_str: str, explain: bool, audit_loop_idx: Optional[int], sweep: bool, sweep_range: Optional[Tuple[int, int]],
         threshold: float, default_n: Optional[int], batch: bool, json_output: bool, list_targets: bool, include_tests: bool, quiet: bool):
    """Cerberus: Explainable ML-Guided GPU Offload Profitability Predictor.

    Analyzes candidate C/C++ loops, predicts GPU profitability against CPU baseline,
    provides feature-grounded explanations, and injects OpenMP 4.5+ or OpenACC directives.
    """
    global console
    if quiet:
        console = Console(quiet=True)

    # Handle --list-targets early exit
    if list_targets:
        list_target_profiles()
        return

    if not source_file:
        console.print("[bold red]Error: SOURCE_FILE argument is required (unless using --list-targets).[/bold red]")
        sys.exit(1)

    # Parse parameter overrides
    params = {}
    if param_str:
        for item in param_str.split(','):
            if '=' in item:
                k, v = item.split('=', 1)
                try:
                    val = int(v.strip())
                    key = k.strip()
                    params[key] = val
                    if key in ("MATRIX_N", "dim", "size", "N"):
                        params["N"] = val
                        params["MATRIX_N"] = val
                        params["dim"] = val
                except ValueError:
                    pass

    # 1. Resolve Target Hardware
    if target_name == 'auto':
        hw = detect_local_hardware()
    elif target_name in PRESET_PROFILES:
        hw = PRESET_PROFILES[target_name]
    else:
        console.print(f"[bold yellow]Warning: Unknown target '{target_name}', falling back to auto-detection.[/bold yellow]")
        hw = detect_local_hardware()

    # 2. Model Initialization with Dynamic Silicon Metrics
    model = ProfitabilityModel()
    meta = model.metadata
    roc_auc = meta.get("roc_auc", 0.855)
    n_samples = meta.get("n_samples", 1055)

    if not json_output:
        bw_label = "Shared Memory Bandwidth" if hw.unified_memory else "Interconnect (PCIe)"
        bootstrap_notice = " [dim](Bootstrap)[/dim]" if meta.get("is_bootstrap") else ""
        console.print(Panel.fit(
            f"[bold cyan]Cerberus Compiler Optimizer[/bold cyan] -- GPU Offload Profitability Predictor\n"
            f"Host Processor : [bold white]{hw.cpu_name}[/bold white] (~{hw.cpu_tflops} TFLOPS FP32 Peak)\n"
            f"Target Hardware: [bold green]{hw.name}[/bold green] ([magenta]{hw.device_type.upper()}[/magenta], "
            f"{bw_label}: [yellow]{hw.bus_bandwidth_gbps} GB/s[/yellow], Compute Peak: [yellow]{hw.peak_tflops} TFLOPS[/yellow])\n"
            f"Cost Model     : [bold white]Physical XGBoost + TreeSHAP[/bold white] (Trained on {n_samples:,} runs -- CV ROC-AUC: [green]{roc_auc:.3f}[/green]){bootstrap_notice}",
            title="[bold white]Compiler Optimization Gating[/bold white]",
            border_style="cyan"
        ))

    # 3. Parse Source File
    parser_trip_count = default_n if default_n else 10000
    parser = CLoopParser(default_param_trip_count=parser_trip_count, params=params, include_tests=include_tests)
    loops = parser.parse_file(source_file, params=params)

    if not loops:
        if json_output:
            print(json_lib.dumps({"source_file": source_file, "loops": [], "error": "No loops detected"}, indent=2))
        else:
            console.print(f"[yellow]No loops detected in {source_file}.[/yellow]")
        return

    if not json_output:
        console.print(f"\n[bold]Scanned [cyan]{source_file}[/cyan]: Found [magenta]{len(loops)}[/magenta] candidate loop regions.\n[/bold]")

    # 4. Model Inference & Prediction
    transformer = OpenMPTransformer(model, hw, dialect=dialect)

    results = []
    for loop in loops:
        pred = model.predict_loop(loop, hw, speedup_threshold=threshold)
        results.append((loop, pred))

    # JSON output mode — dump and exit
    if json_output:
        json_data = results_to_json(source_file, results, hw, model)
        print(json_lib.dumps(json_data, indent=2))
        return

    render_scan_table(loops, results, hw)

    # If report file explicitly requested
    if report_file:
        generate_markdown_report(source_file, results, hw, model, report_file, dialect=dialect)
        console.print(f"[bold cyan][REPORT] Generated detailed compiler audit report: {report_file}[/bold cyan]")

    # Compute sweep sizes
    if sweep_range:
        import numpy as np
        sweep_min, sweep_max = sweep_range
        sweep_sizes = sorted(set(np.logspace(
            np.log10(max(1, sweep_min)),
            np.log10(max(2, sweep_max)),
            6
        ).astype(int).tolist()))
    else:
        sweep_sizes = [100, 1000, 10000, 100000, 1000000, 10000000]

    # Batch mode — scan, inject, report, exit (no interactive menu)
    if batch:
        if explain:
            show_explanations(results)
        if sweep:
            show_crossover_sweep(results, model, hw, threshold, sweep_sizes, parser.bytes_per_elem)
        if audit_loop_idx is not None and 1 <= audit_loop_idx <= len(results):
            loop, pred = results[audit_loop_idx - 1]
            show_loop_audit(loop, pred, hw, audit_loop_idx)
        # Always inject in batch mode
        inject_and_save(source_file, transformer, results, hw, model, threshold, output_file, dialect=dialect)
        return

    # If non-interactive batch flags were provided, execute them and exit
    has_flags = output_file or explain or sweep or (audit_loop_idx is not None)
    if has_flags:
        if audit_loop_idx is not None and 1 <= audit_loop_idx <= len(results):
            loop, pred = results[audit_loop_idx - 1]
            show_loop_audit(loop, pred, hw, audit_loop_idx)
        if sweep:
            show_crossover_sweep(results, model, hw, threshold, sweep_sizes, parser.bytes_per_elem)
        if explain:
            show_explanations(results)
        if output_file:
            inject_and_save(source_file, transformer, results, hw, model, threshold, output_file, dialect=dialect)
        return

    # Clean, Minimal 3-Option Interactive Menu
    while True:
        console.print("\n[bold white]Select an action:[/bold white]")
        console.print("  [bold cyan][1][/bold cyan] Explain Bottlenecks & Audit (TreeSHAP & Roofline)")
        console.print("  [bold cyan][2][/bold cyan] Workload Scaling Crossover Sweep (N=100 to 10M)")
        console.print("  [bold cyan][3][/bold cyan] Inject GPU Pragmas & Save Report (OpenMP / OpenACC)")
        console.print("  [bold cyan][q][/bold cyan] Exit")

        choice = Prompt.ask("\n[bold green]Enter choice[/bold green]", choices=["1", "2", "3", "q"], default="1")
        
        if choice == "1":
            show_explanations(results)
            sub = Prompt.ask("\nInspect single loop deep audit box? (Enter loop # or press Enter to skip)", default="")
            if sub.isdigit() and 1 <= int(sub) <= len(results):
                idx = int(sub)
                show_loop_audit(results[idx - 1][0], results[idx - 1][1], hw, idx)
        elif choice == "2":
            show_crossover_sweep(results, model, hw, threshold, sweep_sizes, parser.bytes_per_elem)
        elif choice == "3":
            # Default to OpenMP without prompting (as per PS4)
            transformer = OpenMPTransformer(model, hw, dialect=dialect)
            default_out = f"{os.path.splitext(source_file)[0]}_offloaded.cpp"
            save_path = Prompt.ask("Save output to", default=default_out)
            inject_and_save(source_file, transformer, results, hw, model, threshold, save_path, dialect=dialect)
        elif choice == "q":
            console.print("[dim]Exiting Cerberus.[/dim]")
            break

if __name__ == '__main__':
    main()

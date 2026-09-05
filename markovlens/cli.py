"""MarkovLens: Explainable ML-Guided GPU Offload Profitability Predictor CLI."""

import os
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from markovlens.parser import CLoopParser
from markovlens.model import ProfitabilityModel
from markovlens.hardware import detect_local_hardware, PRESET_PROFILES
from markovlens.transformer import OpenMPTransformer

console = Console()

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('source_file', type=click.Path(exists=True))
@click.option('-o', '--output', 'output_file', type=click.Path(), default=None,
              help='Path to write optimized source with injected OpenMP target pragmas.')
@click.option('-t', '--target', 'target_name', default='auto',
              help='Target GPU profile (e.g. auto, dgpu_rtx3060, dgpu_rtx4090, igpu_amd_radeon, igpu_intel_iris, egpu_thunderbolt).')
@click.option('-e', '--explain', is_flag=True, default=False,
              help='Print deep TreeSHAP attribution factor breakdown and Roofline model analysis.')
@click.option('--threshold', default=1.1, type=float,
              help='Minimum predicted speedup required to gate GPU offload (default: 1.1x).')
def main(source_file: str, output_file: str, target_name: str, explain: bool, threshold: float):
    """MarkovLens: Explainable ML-Guided GPU Offload Profitability Predictor.

    Analyzes candidate C/C++ loops, predicts GPU profitability against CPU baseline,
    provides feature-grounded explanations, and injects OpenMP 4.5+ target directives.
    """
    # 1. Resolve Target Hardware
    if target_name == 'auto':
        hw = detect_local_hardware()
    elif target_name in PRESET_PROFILES:
        hw = PRESET_PROFILES[target_name]
    else:
        console.print(f"[bold yellow]Warning: Unknown target '{target_name}', falling back to auto-detection.[/bold yellow]")
        hw = detect_local_hardware()

    console.print(Panel.fit(
        f"[bold cyan]MarkovLens v0.1.0[/bold cyan] · GPU Offload Profitability Predictor\n"
        f"Target Hardware: [bold green]{hw.name}[/bold green] ([magenta]{hw.device_type.upper()}[/magenta], "
        f"Interconnect: [yellow]{hw.bus_bandwidth_gbps} GB/s[/yellow], Compute: [yellow]{hw.peak_tflops} TFLOPS[/yellow])",
        title="[bold white]Compiler Optimization Gating[/bold white]",
        border_style="cyan"
    ))

    # 2. Parse Source File
    parser = CLoopParser()
    loops = parser.parse_file(source_file)

    if not loops:
        console.print(f"[yellow]No 'for' loops detected in {source_file}.[/yellow]")
        return

    console.print(f"\n[bold]Scanned [cyan]{source_file}[/cyan]: Found [magenta]{len(loops)}[/magenta] candidate loop regions.\n[/bold]")

    # 3. Model Inference & TreeSHAP
    model = ProfitabilityModel()
    transformer = OpenMPTransformer(model, hw)

    table = Table(title="Loop Region Profitability & Safety Analysis", show_lines=True)
    table.add_column("Region", style="cyan", justify="left")
    table.add_column("Trip / Depth", style="magenta", justify="right")
    table.add_column("Arith. Intensity", justify="right")
    table.add_column("Roofline Bound", justify="right")
    table.add_column("Predicted Speedup", justify="right", style="bold")
    table.add_column("Gating Decision", justify="center")

    results = []

    for i, loop in enumerate(loops, 1):
        pred = model.predict_loop(loop, hw, speedup_threshold=threshold)
        results.append((loop, pred))

        region_title = f"#{i} {loop.function_name}():L{loop.line_start}-{loop.line_end}"
        trip_str = f"{loop.trip_count:,}\n(Depth {loop.nesting_depth})"
        ai_str = f"{loop.arithmetic_intensity:.2f} FLOP/B\n({loop.data_reuse_ratio:.1f}x reuse)"
        roof_str = f"{pred.roofline.attainable_gflops:.1f} GFLOPS\n({'Mem' if pred.roofline.is_memory_bound else 'Compute'}-Bound)"

        if not loop.is_parallel_safe:
            speedup_str = "[red]0.00x[/red]"
            decision_str = "[bold red][REJECT: UNSAFE][/bold red]"
        elif pred.is_profitable:
            speedup_str = f"[bold green]{pred.predicted_speedup:.2f}x[/bold green]"
            decision_str = "[bold green][INJECT OFFLOAD][/bold green]"
        else:
            speedup_str = f"[bold yellow]{pred.predicted_speedup:.2f}x[/bold yellow]"
            decision_str = "[bold yellow][KEEP CPU][/bold yellow]"

        table.add_row(region_title, trip_str, ai_str, roof_str, speedup_str, decision_str)

    console.print(table)

    # 4. Detailed Explanations
    console.print("\n[bold white]Feature-Grounded Compiler Explanations:[/bold white]")
    for i, (loop, pred) in enumerate(results, 1):
        status_color = "green" if pred.is_profitable and loop.is_parallel_safe else ("red" if not loop.is_parallel_safe else "yellow")
        console.print(f"\n[bold cyan]Loop #{i} ({loop.function_name}, Lines {loop.line_start}-{loop.line_end}):[/bold cyan]")
        console.print(f"  [{status_color}]- {pred.primary_explanation}[/{status_color}]")

        if explain and loop.is_parallel_safe:
            console.print("  [white]Top Positive Factors (Pushed towards GPU):[/white]")
            for label, val in pred.top_positive_factors:
                console.print(f"    [green]+{val:.3f} SHAP[/green] : {label}")
            console.print("  [white]Top Negative Factors (Pushed towards CPU):[/white]")
            for label, val in pred.top_negative_factors:
                console.print(f"    [red]{val:.3f} SHAP[/red] : {label}")

    # 5. Transform and Write Output if requested
    if output_file:
        with open(source_file, 'r', encoding='utf-8') as f:
            raw_source = f.read()
        opt_source, _ = transformer.transform_source(raw_source, speedup_threshold=threshold)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(opt_source)
        console.print(f"\n[bold green][SUCCESS] Generated optimized source with OpenMP pragmas: {output_file}[/bold green]")

if __name__ == '__main__':
    main()

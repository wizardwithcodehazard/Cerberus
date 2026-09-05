"""Test AST Loop Feature Extractor."""

import os
from markovlens.parser import CLoopParser

def test_parse_sample_loops():
    test_file = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "synthetic", "sample_loops.c")
    parser = CLoopParser()
    loops = parser.parse_file(test_file)

    print(f"Detected {len(loops)} loops in sample_loops.c\n")
    assert len(loops) == 3, f"Expected 3 loops, found {len(loops)}"

    # Loop 1: vector_add_small
    l1 = loops[0]
    print("=== Loop 1 (Vector Add Small) ===")
    print(f"Function: {l1.function_name}, Lines: {l1.line_start}-{l1.line_end}")
    print(f"Trip count: {l1.trip_count}, Nesting: {l1.nesting_depth}")
    print(f"FLOPs: {l1.total_flops}, Footprint: {l1.memory_footprint_bytes} B")
    print(f"Arithmetic Intensity: {l1.arithmetic_intensity:.4f} FLOP/B")
    print(f"Arrays Read: {l1.arrays_read}, Written: {l1.arrays_written}")
    assert l1.trip_count == 256
    assert l1.nesting_depth == 1

    # Loop 2: matmul_dense
    l2 = loops[1]
    print("\n=== Loop 2 (Dense Matmul) ===")
    print(f"Function: {l2.function_name}, Lines: {l2.line_start}-{l2.line_end}")
    print(f"Trip count: {l2.trip_count:,}, Nesting: {l2.nesting_depth}")
    print(f"FLOPs: {l2.total_flops:,}, Footprint: {l2.memory_footprint_bytes:,} B")
    print(f"Arithmetic Intensity: {l2.arithmetic_intensity:.4f} FLOP/B")
    assert l2.nesting_depth == 3
    assert l2.trip_count == 1024 * 1024 * 1024

    # Loop 3: branchy_stencil
    l3 = loops[2]
    print("\n=== Loop 3 (Branchy Stencil) ===")
    print(f"Function: {l3.function_name}, Lines: {l3.line_start}-{l3.line_end}")
    print(f"Branch Divergence Count: {l3.branch_divergence_count}")
    assert l3.branch_divergence_count >= 1

    print("\n[SUCCESS] All parser tests passed cleanly!")

if __name__ == "__main__":
    test_parse_sample_loops()

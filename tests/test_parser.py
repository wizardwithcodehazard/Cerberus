"""Test AST Loop Feature Extractor."""

import os
from cerberus.parser import CLoopParser

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

    print("\n[SUCCESS] All sample loop parser tests passed cleanly!")

def test_macro_and_while_loop_parsing():
    source = """
    #define N 2048
    #define BLOCK_SIZE (64 * 4)

    void test_macro_loop(double* A, double* B, double* C) {
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i] * 2.0;
        }
    }

    void test_while_loop(float* x, float* y) {
        int idx = 0;
        while (idx < 5000) {
            y[idx] = x[idx] * 3.14f;
            idx++;
        }
    }
    """
    parser = CLoopParser()
    loops = parser.parse_source(source)
    assert len(loops) == 2, f"Expected 2 loops, got {len(loops)}"

    # Check macro expansion
    l1 = loops[0]
    assert l1.trip_count == 2048, f"Expected 2048 trip count from #define, got {l1.trip_count}"
    # Double precision is 8 bytes per element: 3 arrays * 2048 elems * 8 bytes = 49,152 bytes
    assert l1.memory_footprint_bytes >= 49152, f"Expected >= 49152 bytes for double array, got {l1.memory_footprint_bytes}"

    # Check while loop
    l2 = loops[1]
    assert l2.trip_count == 5000, f"Expected 5000 trip count for while loop, got {l2.trip_count}"
    assert l2.is_parallel_safe, "Expected while loop to be parallel safe"

if __name__ == "__main__":
    test_parse_sample_loops()
    test_macro_and_while_loop_parsing()


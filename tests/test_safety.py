"""Test Loop-Carried Dependency & Safety Analysis."""

from markovlens.parser import CLoopParser

def test_parallel_safety_analysis():
    parser = CLoopParser()

    # 1. Embarrassingly Parallel Loop (Safe)
    code_safe = """
    void parallel_vec(float *a, float *b, float *c, int N) {
        for (int i = 0; i < 1024; i++) {
            c[i] = a[i] * 2.0f + b[i];
        }
    }
    """
    loops_safe = parser.parse_source(code_safe)
    assert len(loops_safe) == 1
    print("=== Test 1: Embarrassingly Parallel Loop ===")
    print("Parallel Safe:", loops_safe[0].is_parallel_safe)
    print("Reason:", loops_safe[0].safety_reason)
    assert loops_safe[0].is_parallel_safe is True

    # 2. Sequential Loop-Carried RAW Hazard (Unsafe: A[i] = A[i-1] + ...)
    code_hazard = """
    void prefix_sum(float *A, int N) {
        for (int i = 1; i < 1024; i++) {
            A[i] = A[i - 1] + 2.5f;
        }
    }
    """
    loops_hazard = parser.parse_source(code_hazard)
    assert len(loops_hazard) == 1
    print("\n=== Test 2: Loop-Carried RAW Hazard Loop ===")
    print("Parallel Safe:", loops_hazard[0].is_parallel_safe)
    print("Reason:", loops_hazard[0].safety_reason)
    assert loops_hazard[0].is_parallel_safe is False, "Prefix sum should be flagged as parallel UNSAFE!"

    # 3. Dense MatMul (Data reuse & coalescing)
    code_matmul = """
    void matmul(float *A, float *B, float *C, int N) {
        for (int i = 0; i < 512; i++) {
            for (int j = 0; j < 512; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 512; k++) {
                    sum += A[i * 512 + k] * B[k * 512 + j];
                }
                C[i * 512 + j] = sum;
            }
        }
    }
    """
    loops_mm = parser.parse_source(code_matmul)
    assert len(loops_mm) == 1
    l = loops_mm[0]
    print("\n=== Test 3: Dense Matmul Cache Locality ===")
    print(f"Data Reuse Ratio: {l.data_reuse_ratio:.2f}x (Higher is better for GPU shared memory)")
    print(f"Coalescing Score: {l.coalescing_efficiency:.2f}")
    print(f"Has Reduction: {l.has_reduction} (Var: {l.reduction_var})")
    assert l.data_reuse_ratio > 10.0

    print("\n[SUCCESS] All loop dependency & safety tests passed cleanly!")

if __name__ == "__main__":
    test_parallel_safety_analysis()

"""Test Loop-Carried Dependency & Safety Analysis."""

from cerberus.parser import CLoopParser

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


def test_aliasing_safety():
    """P0-1 regression: read-write aliasing must be flagged as unsafe."""
    parser = CLoopParser()

    # WAR dependency: A[i] reads A[N-i-1] — cannot be parallelised safely
    code_alias = """
    void reverse_in_place(float *A, int N) {
        for (int i = 0; i < N / 2; i++) {
            float tmp = A[i];
            A[i] = A[N - i - 1];
            A[N - i - 1] = tmp;
        }
    }
    """
    loops = parser.parse_source(code_alias)
    assert len(loops) == 1, "Expected 1 loop to be detected"
    print("\n=== Test 4: WAR Aliasing — Reverse In-Place ===")
    print("Parallel Safe:", loops[0].is_parallel_safe)
    print("Reason:", loops[0].safety_reason)
    assert loops[0].is_parallel_safe is False, (
        "Reverse-in-place reads and writes A[] — must be flagged UNSAFE!"
    )

    # Safe: separate input and output arrays — no aliasing
    code_no_alias = """
    void reverse_copy(float *src, float *dst, int N) {
        for (int i = 0; i < N; i++) {
            dst[i] = src[N - i - 1];
        }
    }
    """
    loops_safe = parser.parse_source(code_no_alias)
    assert len(loops_safe) == 1
    print("\n=== Test 5: Reverse-Copy — Separate Arrays (Safe) ===")
    print("Parallel Safe:", loops_safe[0].is_parallel_safe)
    print("Reason:", loops_safe[0].safety_reason)
    assert loops_safe[0].is_parallel_safe is True, (
        "Separate src/dst arrays have no aliasing — must be flagged SAFE!"
    )

    print("\n[SUCCESS] Aliasing safety tests passed!")


if __name__ == "__main__":
    test_parallel_safety_analysis()
    test_aliasing_safety()

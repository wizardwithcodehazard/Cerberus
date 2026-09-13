"""Test LLVM libclang AST Loop Feature Extraction."""

import pytest
from cerberus.clang_parser import ClangASTParser
from cerberus.parser import get_ast_parser

SAMPLE_CODE = """
void matrix_multiply(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}

float parallel_reduction(const float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += data[i] * data[i] * 1.618f;
    }
    return sum;
}
"""

def test_clang_ast_parser_extraction():
    parser = ClangASTParser(params={"N": 512})
    loops = parser.parse_source(SAMPLE_CODE)

    assert len(loops) == 2

    # 1. Matmul loop nest
    matmul = loops[0]
    assert matmul.function_name == "matrix_multiply"
    assert matmul.nesting_depth == 3
    assert matmul.trip_count == 512 * 512 * 512
    assert matmul.is_parallel_safe is True
    assert "A" in matmul.arrays_read
    assert "B" in matmul.arrays_read
    assert "C" in matmul.arrays_written

    # 2. Reduction loop
    reduction = loops[1]
    assert reduction.function_name == "parallel_reduction"
    assert reduction.nesting_depth == 1
    assert reduction.trip_count == 512
    assert reduction.has_reduction is True
    assert reduction.is_parallel_safe is True
    assert "data" in reduction.arrays_read

def test_get_ast_parser_factory():
    clang_p = get_ast_parser(backend="clang")
    assert isinstance(clang_p, ClangASTParser)

    native_p = get_ast_parser(backend="native")
    assert native_p.__class__.__name__ == "CLoopParser"

    auto_p = get_ast_parser(backend="auto")
    assert isinstance(auto_p, ClangASTParser)

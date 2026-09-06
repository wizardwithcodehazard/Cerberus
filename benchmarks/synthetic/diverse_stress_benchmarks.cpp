#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

// ============================================================
// 1. IMAGE SOBEL FILTER (2D Spatial Stencil)
//    Moderate FLOPs, spatial locality, high parallel trip count (1024x1024).
// ============================================================
void image_sobel_filter_2d(const std::vector<float>& src, std::vector<float>& dst, int width, int height) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float gx = -src[idx - width - 1] + src[idx - width + 1]
                       - 2.0f * src[idx - 1] + 2.0f * src[idx + 1]
                       - src[idx + width - 1] + src[idx + width + 1];
            float gy = -src[idx - width - 1] - 2.0f * src[idx - width] - src[idx - width + 1]
                       + src[idx + width - 1] + 2.0f * src[idx + width] + src[idx + width + 1];
            dst[idx] = std::sqrt(gx * gx + gy * gy + 1e-5f);
        }
    }
}

// ============================================================
// 2. MONTE CARLO INTEGRATION / PI (High Compute, Zero Memory Traffic)
//    Massive math calculations per element, reduction on single accumulator.
// ============================================================
float monte_carlo_math_kernel(int N) {
    float pi_accum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(N);
        float term = 4.0f / (1.0f + x * x);
        pi_accum += std::exp(-term * 0.01f) * std::sin(term) * term;
    }
    return pi_accum;
}

// ============================================================
// 3. SPARSE MATRIX-VECTOR (SpMV / Indirect CSR Gather)
//    Indirect column lookups (non-coalesced), low FLOPs per byte.
// ============================================================
void spmv_csr_kernel(const std::vector<float>& values, const std::vector<int>& cols,
                     const std::vector<float>& x, std::vector<float>& y, int num_nonzeros) {
    for (int i = 0; i < num_nonzeros; ++i) {
        y[i] = values[i] * x[cols[i]];
    }
}

// ============================================================
// 4. FIBONACCI / PREFIX SCAN (Loop-Carried RAW Dependency)
//    UNSAFE: A[i] depends directly on A[i-1] from previous iteration.
// ============================================================
void fibonacci_recurrence_unsafe(std::vector<float>& arr, int N) {
    for (int i = 2; i < N; ++i) {
        arr[i] = arr[i - 1] + arr[i - 2] * 0.5f;
    }
}

// ============================================================
// 5. ATTENTION TILE QK^T (3D Nested High Arithmetic Intensity)
//    B x M x N nested contraction, high data reuse, compute bound.
// ============================================================
void attention_score_contraction_3d(const std::vector<float>& Q, const std::vector<float>& K,
                                   std::vector<float>& Out, int B, int M, int N) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float score = 0.0f;
                score += Q[b * M + i] * K[b * N + j] * 0.125f;
                Out[b * (M * N) + i * N + j] = std::tanh(score);
            }
        }
    }
}

// ============================================================
// 6. TINY VECTOR NEGATE (Under-saturated Workload)
//    Trip count is only 64. Launch overhead completely kills GPU.
// ============================================================
void tiny_vector_negate(std::vector<float>& vec) {
    for (int i = 0; i < 64; ++i) {
        vec[i] = -vec[i];
    }
}

int main() {
    std::cout << "Diverse stress benchmark initialized.\n";
    return 0;
}

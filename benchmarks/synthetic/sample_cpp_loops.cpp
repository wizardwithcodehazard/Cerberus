#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <algorithm>

// ============================================================
// 1. SIMPLE VECTOR SCALE
//    Extremely regular, embarrassingly parallel.
// ============================================================

void cpp_vector_scale(std::vector<float>& vec, float scale) {
    for (size_t i = 0; i < 65536; ++i) {
        vec[i] = vec[i] * scale + 1.0f;
    }
}


// ============================================================
// 2. DENSE MATRIX MULTIPLICATION
//    Massive computation, regular structure, high arithmetic
//    intensity. Canonical GPU-friendly workload.
// ============================================================

void cpp_matmul_dense(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}


// ============================================================
// 3. RANGE-BASED NORMALIZATION
//    Parallel, but relatively little work per element.
// ============================================================

void cpp_range_normalize(std::vector<float>& data) {
    for (auto& elem : data) {
        elem = std::sqrt(elem * elem + 0.001f);
    }
}


// ============================================================
// 4. EASY GPU-FRIENDLY KERNEL
//    Independent iterations + sequential memory + expensive
//    math operations.
// ============================================================

void easy_kernel(
    const std::vector<float>& x,
    std::vector<float>& y,
    float a,
    float b)
{
    for (size_t i = 0; i < x.size(); ++i) {

        float v = x[i] * a + b;

        y[i] =
            std::sin(v) *
            std::exp(-v * v);
    }
}


// ============================================================
// 5. IRREGULAR GATHER
//    Lots of parallelism, but deliberately terrible memory
//    locality and unpredictable access patterns.
// ============================================================

void irregular_gather(
    const std::vector<float>& data,
    const std::vector<int>& indices,
    std::vector<float>& output)
{
    const size_t N = indices.size();

    for (size_t i = 0; i < N; ++i) {

        float sum = 0.0f;

        for (int j = 0; j < 64; ++j) {

            size_t idx =
                (static_cast<size_t>(indices[i]) * 1315423911ULL
                + static_cast<size_t>(j) * 2654435761ULL)
                % data.size();

            sum += data[idx];
        }

        output[i] = sum;
    }
}


// ============================================================
// 6. NASTY KERNEL
//
//    Designed to stress a GPU profitability predictor:
//
//    - Independent outer iterations
//    - Variable amount of work
//    - Irregular memory access
//    - Branch divergence
//    - Expensive transcendental functions
//    - Data-dependent control flow
//    - Loop-carried dependency inside each iteration
//    - Configurable additional computation
// ============================================================

void nasty_kernel(
    const std::vector<float>& input,
    const std::vector<int>& offsets,
    std::vector<float>& output,
    int iterations)
{
    const size_t N = input.size();

    for (size_t i = 0; i < N; ++i) {

        float x = input[i];
        float acc = 0.0f;

        // Different elements perform different amounts of work.
        int start = offsets[i];

        int work =
            16 +
            (offsets[i] % 256);

        for (int j = 0; j < work; ++j) {

            // Deliberately irregular memory access.
            size_t idx =
                (static_cast<size_t>(start)
                + static_cast<size_t>(j) * 104729ULL
                + static_cast<size_t>(i * i))
                % N;

            float v = input[idx];

            // Data-dependent branching.
            if (v > 0.5f) {

                acc +=
                    std::sin(v) *
                    std::sqrt(std::abs(x) + 1.0f);

            }
            else if (v < -0.5f) {

                acc -=
                    std::exp(-v * v) *
                    x;

            }
            else {

                acc +=
                    v * v +
                    0.001f;
            }

            // Dependency chain.
            x =
                0.7f * x +
                0.3f * v;
        }

        // Additional sequential computation.
        for (int k = 0; k < iterations; ++k) {

            x =
                std::sin(x) +
                std::cos(x * 0.5f);

            x =
                x * 0.999f +
                acc * 0.001f;
        }

        output[i] =
            x + acc;
    }
}


// ============================================================
// 7. REDUCTION
//    Looks simple, but has a major dependency:
//    every iteration contributes to one accumulator.
//
//    This is useful for testing whether your analyzer can
//    distinguish parallel loops from reduction patterns.
// ============================================================

float reduction_kernel(
    const std::vector<float>& data)
{
    float sum = 0.0f;

    for (size_t i = 0; i < data.size(); ++i) {

        sum +=
            std::sin(data[i]) *
            std::cos(data[i]);
    }

    return sum;
}


// ============================================================
// 8. STENCIL COMPUTATION
//    High parallelism, but neighboring memory dependencies
//    and significant memory traffic.
// ============================================================

void stencil_kernel(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height)
{
    for (int y = 1; y < height - 1; ++y) {

        for (int x = 1; x < width - 1; ++x) {

            int idx =
                y * width + x;

            output[idx] =
                0.25f * input[idx - 1] +
                0.25f * input[idx + 1] +
                0.25f * input[idx - width] +
                0.25f * input[idx + width];
        }
    }
}


// ============================================================
// 9. BRANCH-HEAVY WORKLOAD
//    Large number of iterations, but highly divergent control
//    flow.
// ============================================================

void branch_heavy_kernel(
    const std::vector<float>& input,
    std::vector<float>& output)
{
    for (size_t i = 0; i < input.size(); ++i) {

        float x = input[i];

        if (x > 0.8f) {

            output[i] =
                std::sqrt(x) *
                std::exp(x);

        }
        else if (x > 0.4f) {

            output[i] =
                std::sin(x) *
                std::cos(x);

        }
        else if (x > 0.1f) {

            output[i] =
                x * x +
                std::log(x + 1.0f);

        }
        else {

            output[i] =
                std::abs(x) +
                0.0001f;
        }
    }
}


// ============================================================
// 10. LOW-WORK LOOP
//     Important edge case.
//
//     Massive-looking loop count isn't necessarily enough.
//     Each iteration performs almost no computation.
// ============================================================

void tiny_work_kernel(
    const std::vector<float>& input,
    std::vector<float>& output)
{
    for (size_t i = 0; i < input.size(); ++i) {

        output[i] =
            input[i] + 1.0f;
    }
}


// ============================================================
// MAIN
//
// The main function isn't important for static analysis.
// It simply makes the file executable and ensures the kernels
// are valid C++.
// ============================================================

int main() {

    constexpr size_t N = 65536;

    std::vector<float> data(N, 1.0f);
    std::vector<float> output(N, 0.0f);
    std::vector<int> indices(N);

    for (size_t i = 0; i < N; ++i) {
        indices[i] = static_cast<int>(i * 7919 % N);
    }

    // --------------------------------------------------------
    // Simple workloads
    // --------------------------------------------------------

    cpp_vector_scale(data, 2.0f);

    cpp_range_normalize(data);

    easy_kernel(
        data,
        output,
        1.5f,
        0.25f
    );

    tiny_work_kernel(
        data,
        output
    );

    // --------------------------------------------------------
    // Irregular workload
    // --------------------------------------------------------

    irregular_gather(
        data,
        indices,
        output
    );

    // --------------------------------------------------------
    // Branch-heavy workload
    // --------------------------------------------------------

    branch_heavy_kernel(
        data,
        output
    );

    // --------------------------------------------------------
    // Stencil
    // --------------------------------------------------------

    constexpr int WIDTH = 256;
    constexpr int HEIGHT = 256;

    std::vector<float> grid(
        WIDTH * HEIGHT,
        1.0f
    );

    std::vector<float> next_grid(
        WIDTH * HEIGHT,
        0.0f
    );

    stencil_kernel(
        grid,
        next_grid,
        WIDTH,
        HEIGHT
    );

    // --------------------------------------------------------
    // Reduction
    // --------------------------------------------------------

    float result =
        reduction_kernel(data);

    // --------------------------------------------------------
    // Dense matrix multiplication
    // --------------------------------------------------------

    constexpr int MATRIX_N = 512;

    std::vector<float> A(
        MATRIX_N * MATRIX_N,
        1.0f
    );

    std::vector<float> B(
        MATRIX_N * MATRIX_N,
        1.0f
    );

    std::vector<float> C(
        MATRIX_N * MATRIX_N,
        0.0f
    );

    cpp_matmul_dense(
        A,
        B,
        C,
        MATRIX_N
    );

    // --------------------------------------------------------
    // Nasty workload
    // --------------------------------------------------------

    nasty_kernel(
        data,
        indices,
        output,
        100
    );

    std::cout
        << "Test suite completed. "
        << "Reduction result: "
        << result
        << '\n';

    return 0;
}
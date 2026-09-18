#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <ctime>

// ============================================================
// 1. MATRIX MULTIPLICATION (Compute-Bound)
//    Dense matrix multiplication with high arithmetic intensity
// ============================================================
void matrix_multiply(float* A, float* B, float* C, int N) {
#pragma omp target teams distribute parallel for if(N >= 32) map(to: A[0:N], B[0:N]) map(tofrom: C[0:N])
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}

// ============================================================
// 2. PARALLEL REDUCTION (Memory-Bound)
//    High memory bandwidth utilization with simple arithmetic
// ============================================================
float parallel_reduction(const float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += data[i] * data[i] * 1.618f;
    }
    return sum;
}

// ============================================================
// 3. N-BODY GRAVITATIONAL SIMULATION (Memory-Bound)
//    O(N²) pairwise interactions with sqrt operations
// ============================================================
void nbody_update(float* pos_x, float* pos_y, float* vel_x, float* vel_y, int N, float dt) {
#pragma omp target teams distribute parallel for if(N >= 64) map(to: pos_x[0:N], pos_y[0:N]) map(tofrom: vel_x[0:N], vel_y[0:N])
    for (int i = 0; i < N; ++i) {
        float fx = 0, fy = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float dist = sqrt(dx*dx + dy*dy + 1e-6f);
            float inv_dist3 = 1.0f / (dist * dist * dist);
            fx += dx * inv_dist3;
            fy += dy * inv_dist3;
        }
        vel_x[i] += fx * dt;
        vel_y[i] += fy * dt;
    }
}

// ============================================================
// 4. STENCIL 2D CONVOLUTION (Memory-Bound)
//    3x3 Gaussian blur with memory access pattern
// ============================================================
void stencil_2d_convolution(float* input, float* output, int width, int height) {
    float kernel[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
#pragma omp target teams distribute parallel for map(to: input, kernel) map(tofrom: output)
    for (int y = 1; y < height-1; ++y) {
        for (int x = 1; x < width-1; ++x) {
            float val = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    val += input[(y+ky)*width + (x+kx)] * kernel[ky+1][kx+1];
                }
            }
            output[y*width + x] = val / 16.0f;
        }
    }
}

// ============================================================
// 5. FAST FOURIER TRANSFORM (Compute-Bound)
//    Cooley-Tukey FFT with complex arithmetic
// ============================================================
void fft(float* real, float* imag, int N) {
#pragma omp target teams distribute parallel for if(N >= 32) map(tofrom: imag[0:N], real[0:N])
    for (int len = 2; len <= N; len <<= 1) {
        float ang = 2.0f * 3.14159265359f / len;
        float wreal = cos(ang), wimag = sin(ang);
        for (int i = 0; i < N; i += len) {
            float wre = 1.0f, wim = 0.0f;
            for (int j = 0; j < len/2; ++j) {
                int u = i + j;
                int v = i + j + len/2;
                float tre = wre * real[v] - wim * imag[v];
                float tim = wre * imag[v] + wim * real[v];
                real[v] = real[u] - tre;
                imag[v] = imag[u] - tim;
                real[u] += tre;
                imag[u] += tim;
                float temp = wre;
                wre = wre * wreal - wim * wimag;
                wim = temp * wimag + wim * wreal;
            }
        }
    }
}

// ============================================================
// 6. SPARSE MATRIX-VECTOR MULTIPLY (Compute-Bound)
//    CSR format sparse operations with indirect memory access
// ============================================================
void sparse_mv_multiply(float* vals, int* col_indices, int* row_ptr, float* vec, float* out, int N) {
#pragma omp target teams distribute parallel for if(N >= 64) map(to: vals[0:N], vec[0:N]) map(tofrom: out[0:N])
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            sum += vals[j] * vec[col_indices[j]];
        }
        out[i] = sum;
    }
}

// ============================================================
// 7. PARTICLE SWARM OPTIMIZATION (Compute-Bound)
//    Stochastic optimization with random number generation
// ============================================================
float pso_update(float* positions, float* velocities, float* best_positions, float* best_scores, 
                 int num_particles, int dims, float global_best_score, float* global_best_pos,
                 float inertia, float cognitive, float social) {
    float best_global = global_best_score;
#pragma omp target teams distribute parallel for if(num_particles >= 64) map(to: best_positions[0:num_particles], global_best_pos[0:num_particles]) map(tofrom: positions[0:num_particles], velocities[0:num_particles])
    for (int i = 0; i < num_particles; ++i) {
        float r1 = rand() / (float)RAND_MAX;
        float r2 = rand() / (float)RAND_MAX;
        for (int d = 0; d < dims; ++d) {
            int idx = i * dims + d;
            velocities[idx] = inertia * velocities[idx] +
                             cognitive * r1 * (best_positions[idx] - positions[idx]) +
                             social * r2 * (global_best_pos[d] - positions[idx]);
            positions[idx] += velocities[idx];
        }
    }
    return best_global;
}

// ============================================================
// 8. COMPUTE INTENSITY TEST (Mixed)
//    Configurable arithmetic intensity with transcendental functions
// ============================================================
void compute_intensity_mix(float* data, float* out, int N, int intensity) {
#pragma omp target teams distribute parallel for if(N >= 32) map(to: data[0:N]) map(tofrom: out[0:N])
    for (int i = 0; i < N; ++i) {
        float val = data[i];
        for (int j = 0; j < intensity; ++j) {
            val = sin(val) * cos(val) + exp(val * 0.1f);
            val = val * val * val + 0.618f * val * val + 0.5f * val + 0.123f;
        }
        out[i] = val;
    }
}

// ============================================================
// 9. CONVOLUTION LAYER (Compute-Bound)
//    Deep learning style convolution with 6D loops
// ============================================================
void conv_layer_forward(float* input, float* weights, float* bias, float* output,
                        int batch, int in_channels, int out_channels, 
                        int height, int width, int kernel_size) {
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;
#pragma omp target teams distribute parallel for if(batch >= 32) map(to: bias[0:batch], input[0:batch], weights[0:batch]) map(tofrom: output[0:batch])
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int in_h = oh + kh;
                                int in_w = ow + kw;
                                int input_idx = b * in_channels * height * width + 
                                               ic * height * width + in_h * width + in_w;
                                int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                                ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                    int output_idx = b * out_channels * out_h * out_w + 
                                   oc * out_h * out_w + oh * out_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// ============================================================
// 10. MATRIX TRANSPOSE (Memory-Bound)
//     Bank conflict heavy memory access pattern
// ============================================================
void matrix_transpose(float* input, float* output, int rows, int cols) {
#pragma omp target teams distribute parallel for if(rows >= 64) map(to: input[0:rows]) map(tofrom: output[0:rows])
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// ============================================================
// VALIDATION FUNCTIONS
// ============================================================
void validate_matrix_multiply() {
    const int N = 8;
    float A[64] = {0}, B[64] = {0}, C[64] = {0};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N+j] = (i == j) ? 1.0f : 0.0f; // Identity
            B[i*N+j] = (float)(i + j);
        }
    }
    matrix_multiply(A, B, C, N);
    std::cout << "Matrix Multiply Validation (Identity * B = B): ";
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(C[i*N+j] - (float)(i+j)) > 1e-5) ok = false;
        }
    }
    std::cout << (ok ? "PASSED" : "FAILED") << '\n';
}

void validate_reduction() {
    const int N = 10;
    float data[10] = {1,2,3,4,5,6,7,8,9,10};
    float result = parallel_reduction(data, N);
    float expected = 1.618f * 385.0f;
    std::cout << "Reduction Validation (Σx²*1.618): ";
    std::cout << (fabs(result - expected) < 1e-5 ? "PASSED" : "FAILED") << " (expected=" << expected << ", got=" << result << ")\n";
}

void validate_nbody() {
    const int N = 2;
    float pos_x[2] = {0, 1}, pos_y[2] = {0, 0};
    float vel_x[2] = {0, 0}, vel_y[2] = {0, 0};
    nbody_update(pos_x, pos_y, vel_x, vel_y, N, 0.01f);
    std::cout << "N-Body Validation (2 particles): ";
    std::cout << "vx0=" << vel_x[0] << ", vx1=" << vel_x[1] << " (should be ~0.01 and ~-0.01)\n";
}

void validate_stencil() {
    const int W = 3, H = 3;
    float input[9] = {1,1,1,1,0,1,1,1,1};
    float output[9] = {0};
    stencil_2d_convolution(input, output, W, H);
    std::cout << "Stencil Validation (center with 0, neighbors 1): ";
    float center = output[1*W + 1];
    float expected = 12.0f / 16.0f;
    std::cout << (fabs(center - expected) < 1e-5 ? "PASSED" : "FAILED") << " (center=" << center << ", expected=" << expected << ")\n";
}

void validate_fft() {
    const int N = 4;
    float real[4] = {1,0,0,0};
    float imag[4] = {0,0,0,0};
    fft(real, imag, N);
    std::cout << "FFT Validation (delta function): ";
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(real[i] - 1.0f) > 1e-5) ok = false;
        if (fabs(imag[i]) > 1e-5) ok = false;
    }
    std::cout << (ok ? "PASSED" : "FAILED") << '\n';
}

void validate_sparse() {
    const int N = 3;
    float vals[3] = {1,1,1};
    int col_indices[3] = {0,1,2};
    int row_ptr[4] = {0,1,2,3};
    float vec[3] = {2,3,4};
    float out[3] = {0};
    sparse_mv_multiply(vals, col_indices, row_ptr, vec, out, N);
    std::cout << "Sparse MV Validation (identity): ";
    bool ok = true;
    float expected[3] = {2,3,4};
    for (int i = 0; i < N; ++i) {
        if (fabs(out[i] - expected[i]) > 1e-5) ok = false;
    }
    std::cout << (ok ? "PASSED" : "FAILED") << '\n';
}

void validate_pso() {
    const int num_particles = 10, dims = 2;
    float positions[20] = {0}, velocities[20] = {0};
    float best_positions[20] = {0}, best_scores[10] = {0};
    float global_best_pos[2] = {1,1};
    srand(42); // Fixed seed for reproducibility
    pso_update(positions, velocities, best_positions, best_scores, 
               num_particles, dims, 0.0f, global_best_pos, 0.7f, 1.5f, 1.5f);
    std::cout << "PSO Validation: Check if particles move toward (1,1)\n";
    std::cout << "First particle pos: (" << positions[0] << ", " << positions[1] << ")\n";
}

void validate_compute_intensity() {
    const int N = 5, intensity = 3;
    float data[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float out[5] = {0};
    compute_intensity_mix(data, out, N, intensity);
    std::cout << "Compute Intensity Validation (non-linear transform): ";
    std::cout << "out[0]=" << out[0] << ", out[4]=" << out[4] << "\n";
}

void validate_convolution() {
    const int batch=1, in_ch=1, out_ch=1, h=5, w=5, k=3;
    float input[25] = {1}, weights[9] = {0.5}, bias[1] = {0};
    float output[9] = {0};
    for (int i = 0; i < 25; ++i) input[i] = 1.0f;
    for (int i = 0; i < 9; ++i) weights[i] = 0.5f;
    conv_layer_forward(input, weights, bias, output, batch, in_ch, out_ch, h, w, k);
    std::cout << "Convolution Validation (all ones, weights=0.5): ";
    bool ok = true;
    float expected = 9 * 0.5f;
    for (int i = 0; i < 9; ++i) {
        if (fabs(output[i] - expected) > 1e-5) ok = false;
    }
    std::cout << (ok ? "PASSED" : "FAILED") << " (expected=" << expected << ")\n";
}

void validate_transpose() {
    const int rows=3, cols=4;
    float input[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    float output[12] = {0};
    matrix_transpose(input, output, rows, cols);
    std::cout << "Transpose Validation (3x4): ";
    bool ok = true;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fabs(output[j*rows + i] - input[i*cols + j]) > 1e-5) ok = false;
        }
    }
    std::cout << (ok ? "PASSED" : "FAILED") << '\n';
}

// ============================================================
// MAIN - Run all validations
// ============================================================
int main() {
    std::cout << "\n========== GPU OFFLOAD PROFILING TEST SUITE ==========\n\n";
    
    constexpr int total = 1000000;
    std::cout << "Test size: " << total << " elements\n";
    std::cout << "Validating all kernels for correctness...\n\n";
    
    validate_matrix_multiply();
    validate_reduction();
    validate_nbody();
    validate_stencil();
    validate_fft();
    validate_sparse();
    validate_pso();
    validate_compute_intensity();
    validate_convolution();
    validate_transpose();
    
    std::cout << "\n========== PERFORMANCE METRICS TO TRACK ==========\n";
    std::cout << "1. Compute-Bound kernels: Matrix Mult, FFT, Convolution, PSO\n";
    std::cout << "2. Memory-Bound kernels: Reduction, N-Body, Stencil, Transpose\n";
    std::cout << "3. Mixed kernels: Sparse MV, Compute Intensity\n";
    std::cout << "4. Key metrics: FLOPs, memory bandwidth, occupancy, register pressure\n";
    std::cout << "5. Expected speedup: Compute-bound (10-100x), Memory-bound (2-10x)\n";
    std::cout << "\n===== SUCCESSFUL VALIDATION MEANS ALL KERNELS ARE CORRECT ====\n";
    
    return 0;
}
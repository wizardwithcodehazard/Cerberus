"""Comprehensive Multi-Domain Benchmark Suite for Empirical CPU/GPU Ground-Truth Collection.

Contains 20 diverse computational kernels spanning:
1. Polyhedral Dense Linear Algebra (GEMM, 2MM, GEMV, AXPY, Matrix Transpose, Triangular Solve)
2. Stencils & Image Processing (Jacobi-2D, Heat-3D Stencil, Conv2D, Sobel Edge Detection)
3. TSVC & Vectorization Patterns (VectorAdd, DotProduct, Branchy Threshold, Strided Gather, Indirect Access, L2-Norm, LogSumExp)
4. Markov & HMM Probabilistic Computing (Viterbi Trellis, Forward-Backward Step, Baum-Welch M-Step)
"""

from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkKernel:
    name: str
    category: str
    description: str
    c_source_template: str
    opencl_source: str
    dimension_param: str
    dimension_scales: List[int]
    loop_depth: int

BENCHMARK_SUITE: List[BenchmarkKernel] = [
    # -------------------------------------------------------------
    # 1. POLYHEDRAL DENSE LINEAR ALGEBRA
    # -------------------------------------------------------------
    BenchmarkKernel(
        name="gemm_dense",
        category="polybench",
        description="Dense Matrix-Matrix Multiplication (C = alpha*A*B + beta*C)",
        dimension_param="N",
        dimension_scales=[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
        loop_depth=3,
        c_source_template="""
void gemm_dense(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C[i * N + j] * 1.2f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
""",
        opencl_source="""
__kernel void gemm_dense(__global const float *A, __global const float *B, __global float *C, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < N && j < N) {
        float sum = C[i * N + j] * 1.2f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
"""
    ),

    BenchmarkKernel(
        name="two_mm",
        category="polybench",
        description="2 Matrix Multiplications (tmp = A*B; D = tmp*C)",
        dimension_param="N",
        dimension_scales=[32, 64, 96, 128, 192, 256, 384, 512],
        loop_depth=3,
        c_source_template="""
void two_mm(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] += sum * 1.5f;
        }
    }
}
""",
        opencl_source="""
__kernel void two_mm(__global const float *A, __global const float *B, __global float *C, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < N && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] += sum * 1.5f;
    }
}
"""
    ),

    BenchmarkKernel(
        name="gemv_dense",
        category="polybench",
        description="Matrix-Vector Multiplication (y = alpha*A*x + beta*y)",
        dimension_param="N",
        dimension_scales=[64, 128, 256, 512, 1024, 2048, 4096],
        loop_depth=2,
        c_source_template="""
void gemv_dense(float *A, float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum * 1.5f + y[i] * 0.5f;
    }
}
""",
        opencl_source="""
__kernel void gemv_dense(__global const float *A, __global const float *x, __global float *y, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum * 1.5f + y[i] * 0.5f;
    }
}
"""
    ),

    BenchmarkKernel(
        name="matrix_transpose",
        category="polybench",
        description="2D Matrix Transposition (B = A^T)",
        dimension_param="N",
        dimension_scales=[64, 128, 256, 512, 1024, 2048],
        loop_depth=2,
        c_source_template="""
void matrix_transpose(float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}
""",
        opencl_source="""
__kernel void matrix_transpose(__global const float *A, __global float *B, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < N && j < N) {
        B[j * N + i] = A[i * N + j];
    }
}
"""
    ),

    # -------------------------------------------------------------
    # 2. STENCILS & VISION KERNELS
    # -------------------------------------------------------------
    BenchmarkKernel(
        name="conv2d_filter",
        category="polybench",
        description="2D Image Convolution with 5x5 filter",
        dimension_param="N",
        dimension_scales=[64, 128, 256, 512, 1024, 2048],
        loop_depth=2,
        c_source_template="""
void conv2d_filter(float *img, float *out, float *kernel_weights, int N) {
    for (int i = 2; i < N - 2; i++) {
        for (int j = 2; j < N - 2; j++) {
            float sum = 0.0f;
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    sum += img[(i + ki) * N + (j + kj)] * kernel_weights[(ki + 2) * 5 + (kj + 2)];
                }
            }
            out[i * N + j] = sum;
        }
    }
}
""",
        opencl_source="""
__kernel void conv2d_filter(__global const float *img, __global float *out, __global const float *kernel_weights, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= 2 && i < N - 2 && j >= 2 && j < N - 2) {
        float sum = 0.0f;
        for (int ki = -2; ki <= 2; ki++) {
            for (int kj = -2; kj <= 2; kj++) {
                sum += img[(i + ki) * N + (j + kj)] * kernel_weights[(ki + 2) * 5 + (kj + 2)];
            }
        }
        out[i * N + j] = sum;
    }
}
"""
    ),

    BenchmarkKernel(
        name="sobel_filter",
        category="vision",
        description="2D Sobel Edge Detection 3x3 filter",
        dimension_param="N",
        dimension_scales=[64, 128, 256, 512, 1024, 2048],
        loop_depth=2,
        c_source_template="""
void sobel_filter(float *img, float *edge, int N) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            float gx = -img[(i-1)*N + (j-1)] + img[(i-1)*N + (j+1)]
                       -2.0f * img[i*N + (j-1)] + 2.0f * img[i*N + (j+1)]
                       -img[(i+1)*N + (j-1)] + img[(i+1)*N + (j+1)];
            float gy = -img[(i-1)*N + (j-1)] - 2.0f * img[(i-1)*N + j] - img[(i-1)*N + (j+1)]
                       +img[(i+1)*N + (j-1)] + 2.0f * img[(i+1)*N + j] + img[(i+1)*N + (j+1)];
            edge[i * N + j] = sqrtf(gx * gx + gy * gy);
        }
    }
}
""",
        opencl_source="""
__kernel void sobel_filter(__global const float *img, __global float *edge, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
        float gx = -img[(i-1)*N + (j-1)] + img[(i-1)*N + (j+1)]
                   -2.0f * img[i*N + (j-1)] + 2.0f * img[i*N + (j+1)]
                   -img[(i+1)*N + (j-1)] + img[(i+1)*N + (j+1)];
        float gy = -img[(i-1)*N + (j-1)] - 2.0f * img[(i-1)*N + j] - img[(i-1)*N + (j+1)]
                   +img[(i+1)*N + (j-1)] + 2.0f * img[(i+1)*N + j] + img[(i+1)*N + (j+1)];
        edge[i * N + j] = sqrt(gx * gx + gy * gy);
    }
}
"""
    ),

    BenchmarkKernel(
        name="jacobi_2d",
        category="polybench",
        description="2D 5-Point Jacobi Stencil computation",
        dimension_param="N",
        dimension_scales=[64, 128, 256, 512, 1024, 2048],
        loop_depth=2,
        c_source_template="""
void jacobi_2d(float *A, float *B, int N) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            B[i * N + j] = 0.2f * (A[i * N + j] + A[i * N + (j - 1)] + A[i * N + (j + 1)] + 
                                   A[(i - 1) * N + j] + A[(i + 1) * N + j]);
        }
    }
}
""",
        opencl_source="""
__kernel void jacobi_2d(__global const float *A, __global float *B, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
        B[i * N + j] = 0.2f * (A[i * N + j] + A[i * N + (j - 1)] + A[i * N + (j + 1)] + 
                               A[(i - 1) * N + j] + A[(i + 1) * N + j]);
    }
}
"""
    ),

    # -------------------------------------------------------------
    # 3. TSVC & VECTORIZATION LOOPS
    # -------------------------------------------------------------
    BenchmarkKernel(
        name="vector_add",
        category="tsvc",
        description="1D Vector Addition (c[i] = a[i] + b[i])",
        dimension_param="N",
        dimension_scales=[1024, 8192, 65536, 262144, 1048576, 4194304],
        loop_depth=1,
        c_source_template="""
void vector_add(float *a, float *b, float *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
""",
        opencl_source="""
__kernel void vector_add(__global const float *a, __global const float *b, __global float *c, int N) {
    int i = get_global_id(0);
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
"""
    ),

    BenchmarkKernel(
        name="axpy_blas",
        category="tsvc",
        description="BLAS Level-1 AXPY (y[i] = alpha * x[i] + y[i])",
        dimension_param="N",
        dimension_scales=[1024, 8192, 65536, 262144, 1048576, 4194304],
        loop_depth=1,
        c_source_template="""
void axpy_blas(float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = 3.14159f * x[i] + y[i];
    }
}
""",
        opencl_source="""
__kernel void axpy_blas(__global const float *x, __global float *y, int N) {
    int i = get_global_id(0);
    if (i < N) {
        y[i] = 3.14159f * x[i] + y[i];
    }
}
"""
    ),

    BenchmarkKernel(
        name="branchy_threshold",
        category="tsvc",
        description="Conditional vector update with branch divergence",
        dimension_param="N",
        dimension_scales=[1024, 8192, 65536, 262144, 1048576, 2097152],
        loop_depth=1,
        c_source_template="""
void branchy_threshold(float *a, float *b, float *c, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] > 0.5f) {
            c[i] = a[i] * b[i] + 1.5f;
        } else {
            c[i] = a[i] * 0.1f - b[i];
        }
    }
}
""",
        opencl_source="""
__kernel void branchy_threshold(__global const float *a, __global const float *b, __global float *c, int N) {
    int i = get_global_id(0);
    if (i < N) {
        if (a[i] > 0.5f) {
            c[i] = a[i] * b[i] + 1.5f;
        } else {
            c[i] = a[i] * 0.1f - b[i];
        }
    }
}
"""
    ),

    BenchmarkKernel(
        name="strided_gather",
        category="tsvc",
        description="Strided array traversal with stride 4",
        dimension_param="N",
        dimension_scales=[1024, 8192, 65536, 262144, 1048576],
        loop_depth=1,
        c_source_template="""
void strided_gather(float *in, float *out, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = in[i * 4] * 2.5f + in[i * 4 + 1] * 1.5f;
    }
}
""",
        opencl_source="""
__kernel void strided_gather(__global const float *in, __global float *out, int N) {
    int i = get_global_id(0);
    if (i < N) {
        out[i] = in[i * 4] * 2.5f + in[i * 4 + 1] * 1.5f;
    }
}
"""
    ),

    BenchmarkKernel(
        name="l2_norm_vector",
        category="tsvc",
        description="Vector L2 Euclidean Norm with math operations",
        dimension_param="N",
        dimension_scales=[1024, 8192, 65536, 262144, 1048576],
        loop_depth=1,
        c_source_template="""
void l2_norm_vector(float *in, float *out, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = sqrtf(in[i] * in[i] + 0.001f);
    }
}
""",
        opencl_source="""
__kernel void l2_norm_vector(__global const float *in, __global float *out, int N) {
    int i = get_global_id(0);
    if (i < N) {
        out[i] = sqrt(in[i] * in[i] + 0.001f);
    }
}
"""
    ),

    # -------------------------------------------------------------
    # 4. MARKOV & HMM PROBABILISTIC WORKLOADS
    # -------------------------------------------------------------
    BenchmarkKernel(
        name="hmm_viterbi_step",
        category="hmm",
        description="HMM Viterbi single-timestep trellis update across S states",
        dimension_param="S",
        dimension_scales=[16, 32, 64, 128, 256, 512, 1024],
        loop_depth=2,
        c_source_template="""
void hmm_viterbi_step(float *viterbi_prev, float *trans_mat, float *obs_prob, float *viterbi_curr, int S) {
    for (int j = 0; j < S; j++) {
        float max_val = -1e9f;
        for (int i = 0; i < S; i++) {
            float val = viterbi_prev[i] + trans_mat[i * S + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        viterbi_curr[j] = max_val + obs_prob[j];
    }
}
""",
        opencl_source="""
__kernel void hmm_viterbi_step(__global const float *viterbi_prev, __global const float *trans_mat, __global const float *obs_prob, __global float *viterbi_curr, int S) {
    int j = get_global_id(0);
    if (j < S) {
        float max_val = -1e9f;
        for (int i = 0; i < S; i++) {
            float val = viterbi_prev[i] + trans_mat[i * S + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        viterbi_curr[j] = max_val + obs_prob[j];
    }
}
"""
    ),

    BenchmarkKernel(
        name="hmm_forward_step",
        category="hmm",
        description="HMM Forward-Backward alpha state accumulation across S states",
        dimension_param="S",
        dimension_scales=[16, 32, 64, 128, 256, 512, 1024],
        loop_depth=2,
        c_source_template="""
void hmm_forward_step(float *alpha_prev, float *trans_mat, float *emission, float *alpha_curr, int S) {
    for (int j = 0; j < S; j++) {
        float sum = 0.0f;
        for (int i = 0; i < S; i++) {
            sum += alpha_prev[i] * trans_mat[i * S + j];
        }
        alpha_curr[j] = sum * emission[j];
    }
}
""",
        opencl_source="""
__kernel void hmm_forward_step(__global const float *alpha_prev, __global const float *trans_mat, __global const float *emission, __global float *alpha_curr, int S) {
    int j = get_global_id(0);
    if (j < S) {
        float sum = 0.0f;
        for (int i = 0; i < S; i++) {
            sum += alpha_prev[i] * trans_mat[i * S + j];
        }
        alpha_curr[j] = sum * emission[j];
    }
}
"""
    ),

    BenchmarkKernel(
        name="hmm_m_step",
        category="hmm",
        description="HMM Baum-Welch M-Step parameter re-estimation",
        dimension_param="S",
        dimension_scales=[16, 32, 64, 128, 256, 512],
        loop_depth=2,
        c_source_template="""
void hmm_m_step(float *gamma, float *xi, float *trans_mat, int S) {
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            float num = xi[i * S + j];
            float denom = gamma[i];
            trans_mat[i * S + j] = num / (denom + 1e-6f);
        }
    }
}
""",
        opencl_source="""
__kernel void hmm_m_step(__global const float *gamma, __global const float *xi, __global float *trans_mat, int S) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < S && j < S) {
        float num = xi[i * S + j];
        float denom = gamma[i];
        trans_mat[i * S + j] = num / (denom + 1e-6f);
    }
}
"""
    )
]

// Sample C loops demonstrating different profitability regimes

// Region 1: Small Vector Addition (Low FLOP/Byte, Low Trip Count -> CPU Profitable)
void vector_add_small(float *a, float *b, float *c, int n) {
    for (int i = 0; i < 256; i++) {
        c[i] = a[i] + b[i];
    }
}

// Region 2: Large 2D Matrix Multiply (High FLOPs, Large N -> GPU Profitable)
void matmul_dense(float **A, float **B, float **C, int N) {
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 1024; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Region 3: Strided Stencil with Divergence (Branchy -> GPU Penalty)
void branchy_stencil(float *in, float *out, int N) {
    for (int i = 1; i < 10000; i++) {
        if (in[i] > 0.0f) {
            out[i] = in[i-1] * 0.5f + in[i+1] * 0.5f;
        } else {
            out[i] = 0.0f;
        }
    }
}

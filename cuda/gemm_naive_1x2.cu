__global__ void gemm_naive_1x2_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y, col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (row < m && col < n) {
    double s0 = 0, s1 = 0;
    for (int p = 0; p < k; ++p) {
      double av = a[row * k + p];
      s0 += av * b[p * n + col];
      if (col + 1 < n) {
        s1 += av * b[p * n + col + 1];
      }
    }
    c[row * n + col] = s0;
    if (col + 1 < n) {
      c[row * n + col + 1] = s1;
    }
  }
}

void launch_naive_1x2(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(16, 16), grid((n + 31) / 32, (m + 15) / 16);
  gemm_naive_1x2_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

GpuVersion version_naive_1x2{"cuda-naive-1x2", launch_naive_1x2};

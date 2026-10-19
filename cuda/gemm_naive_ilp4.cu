__global__ void gemm_naive_ilp4_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int p = 0;
    for (; p + 3 < k; p += 4) {
      s0 += a[row * k + p] * b[p * n + col];
      s1 += a[row * k + p + 1] * b[(p + 1) * n + col];
      s2 += a[row * k + p + 2] * b[(p + 2) * n + col];
      s3 += a[row * k + p + 3] * b[(p + 3) * n + col];
    }
    for (; p < k; ++p) {
      s0 += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = (s0 + s1) + (s2 + s3);
  }
}

void launch_naive_ilp4(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(16, 16), grid((n + 15) / 16, (m + 15) / 16);
  gemm_naive_ilp4_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

GpuVersion version_naive_ilp4{"cuda-naive-ilp4", launch_naive_ilp4};

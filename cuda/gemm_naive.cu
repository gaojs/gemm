__global__ void gemm_naive_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    double sum = 0.0;
    for (int p = 0; p < k; ++p) {
      sum += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = sum;
  }
}

void launch_naive(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(16, 16), grid((n + 15) / 16, (m + 15) / 16);
  gemm_naive_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

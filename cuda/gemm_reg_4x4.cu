__global__ void gemm_reg_4x4_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  __shared__ double at[32][32], bt[32][32];
  int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 8 + tx;
  int row = blockIdx.y * 32 + ty * 4, col = blockIdx.x * 32 + tx * 4;
  double s[4][4] = {{0}};
  for (int t = 0; t < (k + 31) / 32; ++t) {
    for (int x = tid; x < 1024; x += 64) {
      int r = x / 32, q = x % 32, ar = blockIdx.y * 32 + r;
      int ac = t * 32 + q, br = t * 32 + r, bc = blockIdx.x * 32 + q;
      at[r][q] = (ar < m && ac < k) ? a[ar * k + ac] : 0;
      bt[r][q] = (br < k && bc < n) ? b[br * n + bc] : 0;
    }
    __syncthreads();
    for (int p = 0; p < 32; ++p) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          s[i][j] += at[ty * 4 + i][p] * bt[p][tx * 4 + j];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (row + i < m && col + j < n) {
        c[(row + i) * n + col + j] = s[i][j];
      }
    }
  }
}

void launch_reg_4x4(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(8, 8), grid((n + 31) / 32, (m + 31) / 32);
  gemm_reg_4x4_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

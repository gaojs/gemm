__global__ void gemm_reg_4x4_64_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  __shared__ double at[64][8], bt[8][64];
  int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  int row = blockIdx.y * 64 + ty * 4, col = blockIdx.x * 64 + tx * 4;
  double s[4][4] = {{0}};
  for (int t = 0; t < (k + 7) / 8; ++t) {
    for (int x = tid; x < 512; x += 256) {
      int ar = x / 8, aq = x % 8, br = x / 64, bq = x % 64;
      int gar = blockIdx.y * 64 + ar, gaq = t * 8 + aq;
      int gbr = t * 8 + br, gbq = blockIdx.x * 64 + bq;
      at[ar][aq] = (gar < m && gaq < k) ? a[gar * k + gaq] : 0;
      bt[br][bq] = (gbr < k && gbq < n) ? b[gbr * n + gbq] : 0;
    }
    __syncthreads();
    for (int p = 0; p < 8; ++p) {
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

void launch_reg_4x4_64(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(16, 16), grid((n + 63) / 64, (m + 63) / 64);
  gemm_reg_4x4_64_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

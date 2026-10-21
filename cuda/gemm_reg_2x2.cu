__global__ void gemm_reg_2x2_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  __shared__ double at[32][32], bt[32][32];
  int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  int row = blockIdx.y * 32 + ty * 2, col = blockIdx.x * 32 + tx * 2;
  double s[2][2] = {{0, 0}, {0, 0}};
  for (int t = 0; t < (k + 31) / 32; ++t) {
    for (int x = tid; x < 1024; x += 256) {
      int r = x / 32, q = x % 32;
      int ar = blockIdx.y * 32 + r, ac = t * 32 + q, br = t * 32 + r, bc = blockIdx.x * 32 + q;
      at[r][q] = (ar < m && ac < k) ? a[ar * k + ac] : 0;
      bt[r][q] = (br < k && bc < n) ? b[br * n + bc] : 0;
    }
    __syncthreads();
    for (int p = 0; p < 32; ++p) {
      double x = at[ty * 2][p], y = at[ty * 2 + 1][p];
      s[0][0] += x * bt[p][tx * 2];
      s[0][1] += x * bt[p][tx * 2 + 1];
      s[1][0] += y * bt[p][tx * 2];
      s[1][1] += y * bt[p][tx * 2 + 1];
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    c[row * n + col] = s[0][0];
  }
  if (row < m && col + 1 < n) {
    c[row * n + col + 1] = s[0][1];
  }
  if (row + 1 < m && col < n) {
    c[(row + 1) * n + col] = s[1][0];
  }
  if (row + 1 < m && col + 1 < n) {
    c[(row + 1) * n + col + 1] = s[1][1];
  }
}

void launch_reg_2x2(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(16, 16), grid((n + 31) / 32, (m + 31) / 32);
  gemm_reg_2x2_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

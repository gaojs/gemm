#define TILE 32

__global__ void gemm_tiled_32_kernel(int m, int n, int k, const double *a, const double *b, double *c) {
  __shared__ double at[TILE][TILE], bt[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
  double sum = 0;
  for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
    int ac = t * TILE + threadIdx.x, br = t * TILE + threadIdx.y;
    at[threadIdx.y][threadIdx.x] = (row < m && ac < k) ? a[row * k + ac] : 0;
    bt[threadIdx.y][threadIdx.x] = (br < k && col < n) ? b[br * n + col] : 0;
    __syncthreads();
    for (int p = 0; p < TILE; ++p) {
      sum += at[threadIdx.y][p] * bt[p][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}

void launch_tiled_32(int m, int n, int k, const double *a, const double *b, double *c) {
  dim3 block(TILE, TILE), grid((n + 31) / 32, (m + 31) / 32);
  gemm_tiled_32_kernel<<<grid, block>>>(m, n, k, a, b, c);
}

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t status__ = (call); \
  if (status__ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void gemm_naive(int m, int n, int k, const double *a, const double *b, double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    double sum = 0.0;
    for (int p = 0; p < k; ++p) sum += a[row * k + p] * b[p * n + col];
    c[row * n + col] = sum;
  }
}

#define TILE 32
__global__ void gemm_tiled(int m, int n, int k, const double *a, const double *b, double *c) {
  __shared__ double at[TILE][TILE];
  __shared__ double bt[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  double sum = 0.0;
  for (int tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
    int a_col = tile * TILE + threadIdx.x;
    int b_row = tile * TILE + threadIdx.y;
    at[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0;
    bt[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0;
    __syncthreads();
    for (int p = 0; p < TILE; ++p) sum += at[threadIdx.y][p] * bt[p][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum;
}

static void fill_matrix(double *x, int count, unsigned int seed) {
  for (int i = 0; i < count; ++i) {
    seed = 1664525u * seed + 1013904223u;
    x[i] = ((double)(seed & 0xffffu) / 32767.5) - 1.0;
  }
}

static void reference(int m, int n, int k, const double *a, const double *b, double *c) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p) sum += a[i * k + p] * b[p * n + j];
      c[i * n + j] = sum;
    }
}

static double max_diff(const double *x, const double *y, int count) {
  double result = 0.0;
  for (int i = 0; i < count; ++i) {
    double diff = fabs(x[i] - y[i]);
    if (diff > result) result = diff;
  }
  return result;
}

static void launch_kernel(int mode, int m, int n, int k, const double *a, const double *b, double *c) {
  if (mode == 0) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    gemm_naive<<<grid, block>>>(m, n, k, a, b, c);
  } else {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_tiled<<<grid, block>>>(m, n, k, a, b, c);
  }
}

static double time_kernel(int mode, int m, int n, int k, const double *a, const double *b, double *c) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch_kernel(mode, m, n, k, a, b, c);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < 5; ++r) launch_kernel(mode, m, n, k, a, b, c);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / 5.0e3;
}

static double time_cublas(cublasHandle_t handle, int m, int n, int k, const double *a, const double *b, double *c) {
  const double alpha = 1.0, beta = 0.0;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                           &alpha, b, n, a, k, &beta, c, n));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < 5; ++r)
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, b, n, a, k, &beta, c, n));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / 5.0e3;
}

static void run_mode(const char *name, int mode, cublasHandle_t handle) {
  printf("version = '%s';\nMY_MMult = [\n", name);
  for (int p = 64; p <= 1280; p += 64) {
    size_t na = (size_t)p * p;
    double *a = (double *)malloc(na * sizeof(double));
    double *b = (double *)malloc(na * sizeof(double));
    double *ref = (double *)malloc(na * sizeof(double));
    double *out = (double *)malloc(na * sizeof(double));
    fill_matrix(a, (int)na, 17u + p);
    fill_matrix(b, (int)na, 31u + p);
    reference(p, p, p, a, b, ref);
    double *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, na * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&db, na * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dc, na * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(da, a, na * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, na * sizeof(double), cudaMemcpyHostToDevice));
    double seconds = mode == 2 ? time_cublas(handle, p, p, p, da, db, dc)
                                : time_kernel(mode, p, p, p, da, db, dc);
    CUDA_CHECK(cudaMemcpy(out, dc, na * sizeof(double), cudaMemcpyDeviceToHost));
    double gflops = 2.0 * p * p * p / seconds * 1.0e-9;
    printf("%d %le %le\\n", p, gflops, max_diff(ref, out, (int)na));
    fflush(stdout);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    free(a); free(b); free(ref); free(out);
  }
  printf("];\n");
}

int main(void) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("# GPU: %s, compute capability %d.%d, %.1f GiB\n", prop.name, prop.major, prop.minor,
         (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  run_mode("cuda-naive", 0, handle);
  run_mode("cuda-tiled-32", 1, handle);
  run_mode("cublas-dgemm", 2, handle);
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}

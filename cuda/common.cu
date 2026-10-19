#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e__ = (call); \
  if (e__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)
#define CUBLAS_CHECK(call) do { \
  cublasStatus_t e__ = (call); \
  if (e__ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, e__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

void gemm(int m, int n, int k, const double *a, const double *b, double *c);

using LaunchFn = void (*)(int, int, int, const double *, const double *, double *);
struct GpuVersion {
  const char *name;
  LaunchFn launch;
};

void fill_matrix(double *x, int count, unsigned int seed) {
  for (int i = 0; i < count; ++i) {
    seed = 1664525u * seed + 1013904223u;
    x[i] = ((double)(seed & 0xffffu) / 32767.5) - 1.0;
  }
}

double max_diff(const double *x, const double *y, int count) {
  double result = 0.0;
  for (int i = 0; i < count; ++i) {
    result = fmax(result, fabs(x[i] - y[i]));
  }
  return result;
}

void print_result_header(const char *version) {
  printf("version = '%s';\nMY_MMult = [\n", version);
}

void print_result_row(int n, double gflops, double error) {
  printf("%d %le %le\n", n, gflops, error);
  fflush(stdout);
}

void print_result_footer() {
  printf("];\n");
}

static double elapsed_seconds(const std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void run_gpu_version(const GpuVersion &version, int min_size, int max_size) {
  print_result_header(version.name);
  for (int n = min_size; n <= max_size; n += 64) {
    size_t bytes = (size_t)n * n * sizeof(double);
    std::vector<double> a((size_t)n * n), b((size_t)n * n), ref((size_t)n * n), out((size_t)n * n);
    fill_matrix(a.data(), n * n, 17u + n);
    fill_matrix(b.data(), n * n, 31u + n);
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    version.launch(n, n, n, da, db, dc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    double total = 0.0;
    for (int repeat = 0; repeat < 5; ++repeat) {
      da = db = dc = nullptr;
      auto start = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaMalloc(&da, bytes));
      CUDA_CHECK(cudaMalloc(&db, bytes));
      CUDA_CHECK(cudaMalloc(&dc, bytes));
      CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
      version.launch(n, n, n, da, db, dc);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(out.data(), dc, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(da));
      CUDA_CHECK(cudaFree(db));
      CUDA_CHECK(cudaFree(dc));
      total += elapsed_seconds(start);
    }
    double seconds = total / 5.0;
    gemm(n, n, n, a.data(), b.data(), ref.data());
    print_result_row(n, 2.0 * n * n * n / seconds * 1.0e-9, max_diff(ref.data(), out.data(), n * n));
  }
  print_result_footer();
}

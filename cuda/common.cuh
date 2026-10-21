#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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

using LaunchFn = void (*)(int, int, int, const double *, const double *, double *);

struct GpuVersion {
  const char *name;
  LaunchFn launch;
};

extern cublasHandle_t benchmark_cublas_handle;

void gemm(int m, int n, int k, const double *a, const double *b, double *c);
void fill_matrix(double *x, int count, unsigned int seed);
double max_diff(const double *x, const double *y, int count);
void print_result_header(const char *version);
void print_result_row(int n, double gflops, double error);
void print_result_footer();
double elapsed_seconds(const std::chrono::steady_clock::time_point &start);
void run_cpu_version(int min_size, int max_size);
void run_gpu_version(const GpuVersion &version, int min_size, int max_size);

cublasHandle_t benchmark_cublas_handle;

void launch_dgemm_cublas(int m, int n, int k, const double *a, const double *b, double *c) {
  const double alpha = 1.0, beta = 0.0;
  CUBLAS_CHECK(cublasDgemm(benchmark_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n));
}

GpuVersion version_cublas{"cublas", launch_dgemm_cublas};

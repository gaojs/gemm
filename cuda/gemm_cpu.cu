void gemm(int m, int n, int k, const double *a, const double *b, double *c) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p) {
        sum += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

void run_cpu_version(int min_size, int max_size) {
  print_result_header("cpu-reference");
  for (int n = min_size; n <= max_size; n += 64) {
    size_t count = (size_t)n * n;
    std::vector<double> a(count), b(count), ref(count), out(count);
    fill_matrix(a.data(), (int)count, 17u + n);
    fill_matrix(b.data(), (int)count, 31u + n);
    gemm(n, n, n, a.data(), b.data(), out.data());
    double total = 0.0;
    for (int repeat = 0; repeat < 5; ++repeat) {
      auto start = std::chrono::steady_clock::now();
      gemm(n, n, n, a.data(), b.data(), out.data());
      total += elapsed_seconds(start);
    }
    gemm(n, n, n, a.data(), b.data(), ref.data());
    double seconds = total / 5.0;
    print_result_row(n, 2.0 * n * n * n / seconds * 1.0e-9, 
      max_diff(ref.data(), out.data(), (int)count));
  }
  print_result_footer();
}

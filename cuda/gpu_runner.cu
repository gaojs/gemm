#include "common.cuh"

#include "common.cu"
#include "gemm_cpu.cu"
#include "gemm_naive.cu"
#include "gemm_naive_ilp4.cu"
#include "gemm_naive_1x2.cu"
#include "gemm_tiled_32.cu"
#include "gemm_reg_2x2.cu"
#include "gemm_reg_4x4.cu"
#include "gemm_reg_4x4_64.cu"
#include "gemm_cublas.cu"

GpuVersion version_naive{"cuda-naive", launch_naive};
GpuVersion version_naive_ilp4{"cuda-naive-ilp4", launch_naive_ilp4};
GpuVersion version_naive_1x2{"cuda-naive-1x2", launch_naive_1x2};
GpuVersion version_tiled_32{"cuda-tiled-32", launch_tiled_32};
GpuVersion version_reg_2x2{"cuda-reg-2x2", launch_reg_2x2};
GpuVersion version_reg_4x4{"cuda-reg-4x4", launch_reg_4x4};
GpuVersion version_reg_4x4_64{"cuda-reg-4x4-64", launch_reg_4x4_64};
GpuVersion version_cublas{"cublas", launch_dgemm_cublas};

int main(int argc, char **argv) {
  int min_size = 64, max_size = 1280;
  if (argc > 2 || (argc == 2 && (sscanf(argv[1], "%d", &min_size) != 1
     || min_size < 64 || min_size > 1280 || min_size % 64))) {
    fprintf(stderr, "usage: %s [N: 64..1280, multiple of 64]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    max_size = min_size;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("# GPU: %s, compute capability %d.%d, %.1f GiB\n", 
    prop.name, prop.major, prop.minor, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  CUBLAS_CHECK(cublasCreate(&benchmark_cublas_handle));
  run_cpu_version(min_size, max_size);
  run_gpu_version(version_naive, min_size, max_size);
  run_gpu_version(version_cublas, min_size, max_size);
  // run_gpu_version(version_naive_ilp4, min_size, max_size);
  // run_gpu_version(version_naive_1x2, min_size, max_size);
  // run_gpu_version(version_tiled_32, min_size, max_size);
  // run_gpu_version(version_reg_2x2, min_size, max_size);
  // run_gpu_version(version_reg_4x4, min_size, max_size);
  // run_gpu_version(version_reg_4x4_64, min_size, max_size);
  CUBLAS_CHECK(cublasDestroy(benchmark_cublas_handle));
  return 0;
}

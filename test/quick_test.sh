#!/bin/bash

echo "=============================================="
echo "      CPU 频率快速测试 (WSL2 专用)"
echo "=============================================="
echo "说明: 使用全部12核心进行压力测试"
echo "      在运行期间，请在 Windows 任务管理器中"
echo "      观察\"性能\"->\"CPU\"->\"当前速度\""
echo "=============================================="

cat > cpu_stress.cpp << 'EOF'
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

#ifdef __GNUC__
#include <x86intrin.h>
#endif

uint64_t avx2_stress(std::atomic<bool>& running) {
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(2.0);
    __m256d c = _mm256_set1_pd(0.0);
    
    uint64_t iterations = 0;
    
    while (running.load()) {
        for (int i = 0; i < 100000; i++) {
            c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
            a = _mm256_add_pd(a, _mm256_set1_pd(0.0001));
            b = _mm256_add_pd(b, _mm256_set1_pd(0.0001));
        }
        iterations++;
    }
    
    alignas(32) double results[4];
    _mm256_store_pd(results, c);
    volatile double dummy = results[0];
    
    return iterations;
}

int main(int argc, char* argv[]) {
    int num_threads = 12;
    int duration_sec = 10;
    
    if (argc > 1) num_threads = std::stoi(argv[1]);
    if (argc > 2) duration_sec = std::stoi(argv[2]);
    
    std::cout << "线程数: " << num_threads << ", 持续时间: " << duration_sec << "秒" << std::endl;
    std::cout << "请观察 Windows 任务管理器中的 CPU 当前速度..." << std::endl;
    
    std::atomic<bool> running(true);
    std::vector<std::thread> threads;
    std::vector<std::atomic<uint64_t>> counters(num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&running, &counters, i]() {
            counters[i].store(avx2_stress(running));
        });
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    
    running.store(false);
    
    for (auto& t : threads) {
        t.join();
    }
    
    uint64_t total_iterations = 0;
    for (int i = 0; i < num_threads; i++) {
        total_iterations += counters[i].load();
    }
    
    uint64_t total_ops = total_iterations * 100000 * 8;
    double gflops = total_ops / (duration_sec * 1e9);
    
    std::cout << "\n完成!" << std::endl;
    std::cout << "总迭代次数: " << total_iterations << std::endl;
    std::cout << "浮点性能: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
    std::cout << "注意: WSL2 的 TSC 频率固定为 2.6GHz，需在 Windows 任务管理器中观察实际频率" << std::endl;
    
    return 0;
}
EOF

echo "正在编译测试程序..."
g++ -std=c++17 -O3 -march=native -mfma -mavx2 -pthread -o cpu_stress cpu_stress.cpp
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi
echo "编译成功!"

echo ""
echo "测试将在 3 秒后开始..."
sleep 3

taskset -c 0-11 ./cpu_stress 12 10

rm -f cpu_stress cpu_stress.cpp

echo ""
echo "=============================================="
echo "测试完成!"
echo "=============================================="
echo ""
echo "如果频率没有达到 4.5 GHz，请执行以下步骤:"
echo "1. 切换到高性能电源计划:"
echo "   powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
echo ""
echo "2. 如果要切回平衡模式:"
echo "   powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e"
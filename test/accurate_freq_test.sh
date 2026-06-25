#!/bin/bash

echo "=============================================="
echo "      CPU 频率精确测试 (WSL2 专用)"
echo "=============================================="
echo "说明: 使用 RDTSC 指令直接测量 CPU 周期"
echo "      在运行期间，请在 Windows 任务管理器中"
echo "      观察\"性能\"->\"CPU\"->\"当前速度\""
echo "=============================================="

cat > accurate_freq.cpp << 'EOF'
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <cmath>

#ifdef __GNUC__
#include <x86intrin.h>
#endif

uint64_t rdtsc() {
#ifdef __GNUC__
    return __rdtsc();
#else
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#endif
}

void stress_loop(std::atomic<bool>& running) {
    volatile double x = 0.0;
    volatile double result = 0.0;
    
    while (running.load()) {
        for (int i = 0; i < 100000; i++) {
            result += sin(x) * cos(x + 1.0);
            x += 0.0001;
        }
    }
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
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(stress_loop, std::ref(running));
    }
    
    uint64_t start_tsc = rdtsc();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t end_tsc = rdtsc();
    
    running.store(false);
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double elapsed_s = elapsed_ns / 1e9;
    
    uint64_t tsc_diff = end_tsc - start_tsc;
    double tsc_freq_ghz = static_cast<double>(tsc_diff) / elapsed_ns;
    
    std::cout << "\n测试结果:" << std::endl;
    std::cout << "时间: " << std::fixed << std::setprecision(3) << elapsed_s << " 秒" << std::endl;
    std::cout << "TSC 周期数: " << tsc_diff << std::endl;
    std::cout << "TSC 频率: " << std::fixed << std::setprecision(3) << tsc_freq_ghz << " GHz" << std::endl;
    std::cout << "\n注意: WSL2 的 TSC 频率是固定的(constant_tsc)" << std::endl;
    std::cout << "请在 Windows 任务管理器中观察实际 CPU 速度" << std::endl;
    
    return 0;
}
EOF

echo "正在编译测试程序..."
g++ -std=c++17 -O3 -march=native -pthread -o accurate_freq accurate_freq.cpp
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi
echo "编译成功!"

echo ""
echo "测试将在 3 秒后开始..."
sleep 3

taskset -c 0-11 ./accurate_freq 12 10

rm -f accurate_freq accurate_freq.cpp
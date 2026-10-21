#!/bin/bash

echo "=============================================="
echo "      CPU 频率全面测试脚本 v3 (WSL2 专用)"
echo "=============================================="
echo "说明: 使用简单浮点运算(加法+乘法)进行压力测试"
echo "      通过比较运算次数来判断 CPU 实际运行频率"
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
    int num_threads = 1;
    int duration_sec = 10;
    
    if (argc > 1) num_threads = std::stoi(argv[1]);
    if (argc > 2) duration_sec = std::stoi(argv[2]);
    
    std::cout << "线程数: " << num_threads << ", 持续时间: " << duration_sec << "秒" << std::endl;
    
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
    
    std::cout << "完成! 总迭代次数: " << total_iterations << std::endl;
    std::cout << "总运算次数: " << total_ops << std::endl;
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

run_test() {
    local cores=$1
    local desc=$2
    local expected_threads=$3
    
    echo ""
    echo "----------------------------------------------"
    echo "测试: $desc"
    echo "绑定核心: $cores"
    echo "预期线程数: $expected_threads"
    echo "请观察 Windows 任务管理器中的 CPU 当前速度..."
    
    taskset -c $cores ./cpu_stress $expected_threads 10
    
    echo "测试完成!"
    echo ""
}

echo ""
echo "准备开始测试..."
echo "请打开 Windows 任务管理器，观察 CPU 当前速度"
echo "测试将在 5 秒后开始..."
sleep 5

echo ""
echo "========== 阶段1: 单核心测试 =========="
run_test "0" "单核心 P核 (CPU0)" 1
run_test "2" "单核心 P核 (CPU2)" 1
run_test "4" "单核心 P核 (CPU4)" 1
run_test "6" "单核心 P核 (CPU6)" 1
run_test "8" "单核心 E核 (CPU8)" 1
run_test "10" "单核心 E核 (CPU10)" 1

echo ""
echo "========== 阶段2: 双核心测试 =========="
run_test "0,2" "双 P核" 2
run_test "8,10" "双 E核" 2

echo ""
echo "========== 阶段3: 四核心测试 =========="
run_test "0,2,4,6" "四 P核" 4

echo ""
echo "========== 阶段4: 全部核心测试 =========="
run_test "0-11" "全部12核心" 12

echo ""
echo "=============================================="
echo "全部测试完成!"
echo "=============================================="
echo ""
echo "分析结果:"
echo "1. 单核心 P核测试时，GFLOPS 应接近 36.8 (4.6GHz × 8)"
echo "2. 单核心 E核测试时，GFLOPS 应接近 27.2 (3.4GHz × 8)"
echo "3. 如果 GFLOPS 远低于预期，说明 CPU 没有达到睿频"
echo ""
echo "如果频率始终停留在 1.5-2.1GHz，请检查:"
echo "- Windows 电源计划是否设置为\"高性能\""
echo "- 笔记本是否插电运行"
echo "- BIOS 中是否启用了 Turbo Boost"
echo "- 散热是否良好（温度过高会降频）"

rm -f cpu_stress cpu_stress.cpp
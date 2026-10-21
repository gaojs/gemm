#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <fstream>

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

void measure_idle_freq(int iterations = 100) {
    std::cout << "\n=== 测量空闲频率 ===" << std::endl;
    
    for (int i = 0; i < iterations; i++) {
        uint64_t start_tsc = rdtsc();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t end_tsc = rdtsc();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        uint64_t tsc_diff = end_tsc - start_tsc;
        
        double freq_ghz = static_cast<double>(tsc_diff) / duration;
        
        if (i % 10 == 0) {
            std::cout << "迭代 " << std::setw(3) << i << ": " 
                      << std::fixed << std::setprecision(3) << freq_ghz 
                      << " GHz" << std::endl;
        }
    }
}

void measure_stress_freq(int duration_ms = 1000) {
    std::cout << "\n=== 测量满载频率（睿频）===" << std::endl;
    
    std::vector<double> results;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_tsc = rdtsc();
    
    double dummy = 0.0;
    int iterations = 0;
    
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - start_time).count() < duration_ms) {
        for (int j = 0; j < 1000000; j++) {
            dummy += std::sin(j) * std::cos(j);
        }
        iterations++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t end_tsc = rdtsc();
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    uint64_t tsc_diff = end_tsc - start_tsc;
    
    double freq_ghz = static_cast<double>(tsc_diff) / duration;
    
    std::cout << "满载频率: " << std::fixed << std::setprecision(3) << freq_ghz << " GHz" << std::endl;
    std::cout << "迭代次数: " << iterations << std::endl;
    std::cout << "TSC周期数: " << tsc_diff << std::endl;
    std::cout << "耗时: " << duration / 1000000.0 << " ms" << std::endl;
}

void measure_multiplier() {
    std::cout << "\n=== 测量倍频 ===" << std::endl;
    
    std::vector<double> frequencies;
    
    for (int i = 0; i < 50; i++) {
        uint64_t start_tsc = rdtsc();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t end_tsc = rdtsc();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        double freq_ghz = static_cast<double>(end_tsc - start_tsc) / duration;
        
        frequencies.push_back(freq_ghz);
    }
    
    std::sort(frequencies.begin(), frequencies.end());
    
    double min_freq = frequencies.front();
    double max_freq = frequencies.back();
    double avg_freq = 0;
    for (double f : frequencies) avg_freq += f;
    avg_freq /= frequencies.size();
    
    std::cout << "最小频率: " << std::fixed << std::setprecision(3) << min_freq << " GHz" << std::endl;
    std::cout << "最大频率: " << std::fixed << std::setprecision(3) << max_freq << " GHz" << std::endl;
    std::cout << "平均频率: " << std::fixed << std::setprecision(3) << avg_freq << " GHz" << std::endl;
    
    const double bus_freq = 0.1;
    std::cout << "\n假设总线频率: " << bus_freq << " GHz" << std::endl;
    std::cout << "最小倍频: " << std::fixed << std::setprecision(1) << min_freq / bus_freq << "x" << std::endl;
    std::cout << "最大倍频: " << std::fixed << std::setprecision(1) << max_freq / bus_freq << "x" << std::endl;
    std::cout << "平均倍频: " << std::fixed << std::setprecision(1) << avg_freq / bus_freq << "x" << std::endl;
}

void print_cpu_info() {
    std::cout << "\n=== CPU 信息 ===" << std::endl;
    
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos ||
                line.find("cpu MHz") != std::string::npos ||
                line.find("cache size") != std::string::npos ||
                line.find("cpu cores") != std::string::npos ||
                line.find("siblings") != std::string::npos) {
                std::cout << line << std::endl;
            }
        }
        cpuinfo.close();
    }
    
    std::cout << "\n=== 系统时钟信息 ===" << std::endl;
    std::ifstream sysfreq("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies");
    if (sysfreq.is_open()) {
        std::string line;
        std::getline(sysfreq, line);
        std::cout << "可用频率: " << line << std::endl;
        sysfreq.close();
    }
    
    std::ifstream maxfreq("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (maxfreq.is_open()) {
        std::string line;
        std::getline(maxfreq, line);
        std::cout << "硬件最大频率: " << std::stoi(line) / 1000.0 << " GHz" << std::endl;
        maxfreq.close();
    }
    
    std::ifstream minfreq("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq");
    if (minfreq.is_open()) {
        std::string line;
        std::getline(minfreq, line);
        std::cout << "硬件最小频率: " << std::stoi(line) / 1000.0 << " GHz" << std::endl;
        minfreq.close();
    }
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "          CPU 频率测试工具" << std::endl;
    std::cout << "================================================" << std::endl;
    
    print_cpu_info();
    measure_idle_freq();
    measure_stress_freq();
    measure_multiplier();
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "                    测试完成" << std::endl;
    std::cout << "================================================" << std::endl;
    
    return 0;
}

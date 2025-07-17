% Indicate the number of floating point operations that can be executed
% per clock cycle. 指示每个时钟周期可以执行的浮点运算数量。
% 我们物理主机支持 AVX-512：nflops_per_cycle = 16

nflops_per_cycle = 16; 

%
% Indicate the number of processors being used (in case you are using a
% multicore or SMP)指出正在使用的处理器数量（如果您使用的是%多核或SMP）
% 我们仅测试单核优化的性能

nprocessors = 1;

%
% Indicate the clock speed of the processor.  On a Linux machine this info
% can be found in the file /proc/cpuinfo
% 指示处理器的时钟速度。在Linux机器上，此信息可以在/proc/cpuinfo文件中找到
% Note: some processors have a "turbo boost" mode, which increases
% the peak clock rate...
% 注意：一些处理器具有“涡轮增压”（turbo boost）模式，可以提高峰值时钟速率。
% 我们物理主机（lscpu | grep MHz），结果是：CPU max MHz: 3200.0000

GHz_of_processor = 3.2;

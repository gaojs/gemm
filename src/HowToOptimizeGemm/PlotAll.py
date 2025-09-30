# 告诉 matplotlib使用 ​​Agg 后端​​，它用于生成图片文件（如 PNG），
# 完全不需要图形显示系统，因此，就不会报如下的错误了：
# Authorization required, but no authorization protocol specified
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# Indicate the number of floating point operations that can be executed
# per clock cycle. 我们物理主机支持 AVX-512：nflops_per_cycle = 16 
nflops_per_cycle = 16

# Indicate the number of processors being used (in case you are using a
# multicore or SMP). 我们仅测试单核优化的性能
nprocessors = 1

# Indicate the clock speed of the processor.  On a Linux machine this info
# can be found in the file /proc/cpuinfo
#
# Note: some processors have a "turbo boost" mode, which increases
# the peak clock rate... 
# 我们物理主机（lscpu | grep MHz），结果是：CPU max MHz: 3200.0000
GHz_of_processor = 3.2


class Parser:
    def __init__(self, file_name) -> None:
        self.attrs = {}
        with open(file_name) as file:
            self.toks = file.read().split()
            self.toksi = 0
            file.close()
            self.attrs = self.parse()

    def next(self):
        tok = self.toks[self.toksi]
        self.toksi += 1
        return tok

    def get_var_name(self):
        return self.next()

    def get_symbol(self, sym):
        tok = self.next()
        assert(tok == sym)
        return tok

    def get_value(self):
        value = None
        tok = self.next()
        if tok == '[':
            # list
            value = []
            tok = self.next()
            while not tok.startswith(']'):
                value.append(float(tok))
                tok = self.next()
        elif tok.startswith("'"):
            value = tok[1:-2]
        
        assert value != None
        return value
    
    def parse(self):
        res = {}
        while self.toksi < len(self.toks):
            var = self.get_var_name()
            self.get_symbol('=')
            val = self.get_value()
            res[var] = val
        return res
    
    def __getattr__(self, name):
        return self.attrs[name]

old = Parser("output_old.m")
new = Parser("output_new.m")

#print(old)
#print(new)

old_data = np.array(old.MY_MMult).reshape(-1, 3)
new_data = np.array(new.MY_MMult).reshape(-1, 3)

max_gflops = nflops_per_cycle * nprocessors * GHz_of_processor;

fig, ax = plt.subplots()
ax.plot(old_data[:,0], old_data[:,1], 'bo-.', label='old:' + old.version)
ax.plot(new_data[:,0], new_data[:,1], 'r-*', label='new:' + new.version)

ax.set(xlabel='m = n = k', ylabel='GFLOPS/sec.',
       title="OLD = {}, NEW = {}".format(old.version, new.version))
ax.grid()
ax.legend()

ax.set_xlim([old_data[0,0], old_data[-1,0]])
ax.set_ylim([0, max_gflops])

fig.savefig("cmp_" + new.version + ".png")
# plt.show()
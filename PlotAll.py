import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

nflops_per_cycle = 16
nprocessors = 1
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


if len(sys.argv) >= 3:
    old_file = sys.argv[1]
    new_file = sys.argv[2]
else:
    old_file = "output_old.m"
    new_file = "output_new.m"

old = Parser(old_file)
new = Parser(new_file)

old_data = np.array(old.MY_MMult).reshape(-1, 3)
new_data = np.array(new.MY_MMult).reshape(-1, 3)

max_gflops = nflops_per_cycle * nprocessors * GHz_of_processor

fig, ax = plt.subplots()
ax.plot(old_data[:,0], old_data[:,1], 'bo-.', label='old:' + old.version)
ax.plot(new_data[:,0], new_data[:,1], 'r-*', label='new:' + new.version)

ax.set(xlabel='m = n = k', ylabel='GFLOPS',
       title="OLD = {}, NEW = {}".format(old.version, new.version))
ax.grid()
ax.legend()

ax.set_xlim([old_data[0,0], old_data[-1,0]])
ax.set_ylim([0, max_gflops])

filename = "compare_{}_{}.png".format(old.version, new.version)
if len(sys.argv) >= 4:
    filename = sys.argv[3]
fig.savefig(filename, dpi=100, bbox_inches='tight')
print("Image saved as: {}".format(filename))
plt.close(fig)

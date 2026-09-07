import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = sys.argv[1] if len(sys.argv) > 1 else 'fig-gpu/gpu_results.m'
out = sys.argv[2] if len(sys.argv) > 2 else 'fig-gpu/gpu_comparison.png'
text = open(source).read()
series = {}
for name, body in re.findall(r"version = '([^']+)'\s*;\s*MY_MMult = \[([^]]+)\];", text, re.S):
    rows = []
    for row in body.strip().splitlines():
        fields = row.split()
        if len(fields) >= 3:
            rows.append((float(fields[0]), float(fields[1])))
    series[name] = rows

fig, ax = plt.subplots(figsize=(8, 5))
for name, rows in series.items():
    x, y = zip(*rows)
    ax.plot(x, y, marker='o', label=name)
ax.set_xlabel('m = n = k')
ax.set_ylabel('GFLOPS')
ax.set_title('GPU GEMM on NVIDIA A100')
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=140)
print(f'Image saved as: {out}')

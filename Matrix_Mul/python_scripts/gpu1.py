import matplotlib.pyplot as plt

mem_fps = []
times = []

plt.figure(figsize=(15, 5))
plt.xlabel("Memory footprint in KB", fontsize=16)

plt.ylabel("Time spent in ms", fontsize=16)
plt.title("Performance for GPU v1", fontsize=20)

with open("../gpu_sim/gpu1.txt", "r") as f:
    for line in f.readlines():
        line_tokens = line.split()
        if len(line_tokens) == 4:
            mem_fps.append(float(line_tokens[0]))
        elif line_tokens[0] == "GPU":
            avg_str = line_tokens[5]
            avg = float(avg_str[:-2])
            if avg_str[-2:] == "us":
                avg = avg * 0.001
            elif avg_str[-2:] == "ms":
                pass
            elif avg_str[-1:] == "s":
                avg *= 1000
            times.append(avg)

plt.semilogy(mem_fps, times, marker='o', label="GPU v1", linewidth=2.5)
times.clear()

with open("../cpu_sim/cpu_vs_gpu1.txt", "r") as f:
    for line in f.readlines():
        tokens = line.split()
        if len(tokens) == 1:
            times.append(float(tokens[0]) * 1000)

plt.plot(mem_fps, times, marker='s', label="CPU 12 threads", linewidth=2.5)


plt.legend()
plt.tight_layout()
plt.savefig("gpu1_vs_cpu.png")
plt.show()

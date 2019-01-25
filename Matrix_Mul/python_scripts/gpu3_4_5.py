import matplotlib.pyplot as plt

mem_fps = []
times = []
programs = ["gpu2", "gpu3", "gpu4", "gpu5"]
markers = ['x', 'o', 's', '+']

plt.figure(figsize=(13, 8))
plt.xlabel("Memory footprint in KB")
plt.ylabel("Time spent in ms")
plt.title("Performance for various GPU versions")

for i in range(len(programs)):
    program = programs[i]
    marker = markers[i]
    with open("%s.txt" % program, "r") as f:
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

    plt.semilogy(mem_fps[:19], times, marker=marker, label="GPU v%s" % program[-1])
    times.clear()

times.clear()

with open("../cpu.txt", "r") as f:
    for line in f.readlines():
        tokens = line.split()
        if len(tokens) == 1:
            times.append(float(tokens[0]) * 1000)

plt.semilogy(mem_fps[:19], times, marker='d', label="CPU 12 threads")

times.clear()
with open("gpu_lib.txt", "r") as f:
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

plt.semilogy(mem_fps[:19], times, marker='*', label="CUBLAS")

plt.legend(fontsize="x-large")
plt.savefig("gpu2+_vs_cpu.png")
plt.show()
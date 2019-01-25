import matplotlib.pyplot as plt

mem_fps = []
mflops = []
time_percents = []
programs = ["gpu2", "gpu3", "gpu4", "gpu5"]
markers = ['x', 'o', 's', '+']

plt.figure(figsize=(13, 8))
plt.xlabel("Memory footprint in KB")
plt.ylabel("Gflop/s")
plt.title("Performance for various GPU versions")

for i in range(len(programs)):
    program = programs[i]
    marker = markers[i]

    with open("%s.txt" % program, "r") as f:
        for line in f.readlines():
            line_tokens = line.split()

            if len(line_tokens) == 4:
                mem_fps.append(float(line_tokens[0]))
                mflops.append(float(line_tokens[1]) / 1000.0)
            elif line_tokens[0] == "GPU":
                time_percents.append(float(line_tokens[2][:-1]))
    for j in range(len(mflops)):
        mflops[j] = mflops[j] / (time_percents[j] / 100)
    plt.plot(mem_fps[:19], mflops, marker=marker, label="GPU v%s" % program[-1])
    mflops.clear()
    time_percents.clear()

mflops.clear()
time_percents.clear()

with open("../cpu.txt", "r") as f:
    for line in f.readlines():
        tokens = line.split()
        iterations = 1
        if len(tokens) == 4:
            if float(tokens[0]) <= 153600:
                iterations = 500
            elif float(tokens[0]) <= 614400:
                iterations = 10
            else:
                iterations = 2
            mflops.append(float(tokens[1]) * iterations / 1000.0)


plt.plot(mem_fps[:19], mflops, marker='d', label="CPU 12 threads")

mflops.clear()
time_percents.clear()
with open("gpu_lib.txt", "r") as f:
    for line in f.readlines():
        line_tokens = line.split()
        if len(line_tokens) == 4:
            mem_fps.append(float(line_tokens[0]))
            mflops.append(float(line_tokens[1]) / 1000.0)
        elif line_tokens[0] == "GPU":
            time_percents.append(float(line_tokens[2][:-1]))

for j in range(len(mflops)):
    mflops[j] = mflops[j] / (time_percents[j] / 100)
plt.plot(mem_fps[:19], mflops, marker='*', label="CUBLAS")

plt.axhline(y=7065, color="black", label="GPU Limit")
plt.axhline(y=998, color="gray", label="CPU Limit")
plt.legend(fontsize="x-large")
plt.savefig("gpu2+_vs_cpu_mflops.png")
plt.show()

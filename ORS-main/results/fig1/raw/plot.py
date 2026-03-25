import matplotlib.pyplot as plt

# ================== Data ==================
configs = ["original-cpu", "original-gpu", "1cpu", "1gpu", "3cpu", "3gpu", "1cpu+1gpu"]
throughput = [2557.32, 7632.27, 942.428, 866.149, 1339.03, 997.853, 904.441]
normalized = [x / throughput[0] for x in throughput]  # normalized by original-cpu

# ================== Style Config ==================
plt.rcParams.update({
    "font.family": "Times New Roman",   # Comment this line if you prefer sans-serif
    "font.size": 16,
    "figure.figsize": (10, 4.8),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5
})

# ================== Plot 1: Original Throughput ==================
fig1, ax1 = plt.subplots()
ax1.bar(configs, throughput)
ax1.axhline(throughput[0], linestyle="--", linewidth=1.5)   # dashed baseline
ax1.set_ylabel("Requests per second (req/s)")
ax1.set_title("Original Throughput")
ax1.set_xticklabels(configs, rotation=30, ha="right")
fig1.tight_layout()

# Save to files
fig1.savefig("throughput_original.pdf")
fig1.savefig("throughput_original.svg")
fig1.savefig("throughput_original.png", dpi=300)

# ================== Plot 2: Normalized Throughput ==================
fig2, ax2 = plt.subplots()
ax2.bar(configs, normalized)
ax2.axhline(1.0, linestyle="--", linewidth=1.5)   # dashed baseline at normalized CPU=1
ax2.set_ylabel("Normalized Throughput")
ax2.set_title("Normalized Throughput")
ax2.set_xticklabels(configs, rotation=30, ha="right")
fig2.tight_layout()

# Save to files
fig2.savefig("throughput_normalized.pdf")
fig2.savefig("throughput_normalized.svg")
fig2.savefig("throughput_normalized.png", dpi=300)

print("✅ All figures saved: PDF + SVG + PNG(300dpi)")


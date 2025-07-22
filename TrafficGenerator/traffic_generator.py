#!/usr/bin/env python3
"""
traffic_generator.py

Generates a stream of traffic flows with specified distributions,
then expands each flow into a sequence of packets based on a fixed
packet size and the flow's rate. All packets are timestamped in seconds
and saved as CSV streams. Output CSVs and plots go into an output directory
to avoid permission issues.
"""

import random
import bisect
import csv
import os
import math
from dataclasses import dataclass

# === Configuration ===
# Pareto rate distribution parameters
PARETO_SHAPE = 1.5                      # Shape parameter α (must be >1)
PARETO_MEAN = 2.5e9                     # Desired mean rate in bps
MAX_RATE = 40e9                         # Truncate rates above this value (bps)

# Target aggregate incoming rate on link (bps)
TARGET_BITRATE = 1.6e12                 # 1.6 Tbps

# Per-flow application mix
TYPE_NAMES = ["type A", "type B"]     # Flow type labels

# Destination ranges per type
DEST_RANGE_A = (2000, 100_000_000)       # Dest IDs for type A
DEST_RANGE_B = (1, 2000)                 # Dest IDs for type B

# Compute mean per-flow rate for each type (bps)
MEAN_RATE_A = PARETO_MEAN * 4.0          # Type A mean (4×)
MEAN_RATE_B = PARETO_MEAN / 4.0          # Type B mean (¼×)
AVG_FLOW_RATE = 0.5 * (MEAN_RATE_A + MEAN_RATE_B)  # 50:50 mix

# Mean inter-arrival between flows (seconds) to achieve TARGET_BITRATE
MEAN_INTERARRIVAL = AVG_FLOW_RATE / TARGET_BITRATE  # in seconds

# Flow and packet parameters
SIZE_CDF_CSV = "SizeDistributions/FB_Hadoop_Inter_Rack_FlowCDF.csv"  # Size CDF path
NUM_FLOWS = 10                           # Number of flows to generate
PACKET_BYTES = 1500                      # Payload per packet in bytes

# Output directories and files
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FLOW_CSV = os.path.join(OUTPUT_DIR, "flows.csv")
PACKET_CSV = os.path.join(OUTPUT_DIR, "packets.csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# == Helper: truncated Pareto scale computation ==
def compute_truncated_scale(alpha: float, desired_mean: float, M: float) -> float:
    low, high = 1e-9, M
    for _ in range(100):
        mid = 0.5 * (low + high)
        p_tail = (mid / M) ** alpha
        E_cont = alpha * mid / (alpha - 1) * (1 - (mid / M) ** (alpha - 1))
        mean_trunc = E_cont + M * p_tail
        if mean_trunc > desired_mean:
            high = mid
        else:
            low = mid
    return mid

PARETO_SCALE = compute_truncated_scale(PARETO_SHAPE, PARETO_MEAN, MAX_RATE)
print(f"[*] Computed Pareto x_m = {PARETO_SCALE:.2e}, max_rate = {MAX_RATE:.2e}")

# Sampling config for plotting
SAMPLE_COUNT = 1_000_000
BIN_COUNT = 100

# Try import plotting libs
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    print(f"[*] Plotting disabled ({e}).")

@dataclass
class Flow:
    start_time: float  # seconds
    size: int          # bytes
    rate: float        # bps
    destination: int
    type: str

@dataclass
class Packet:
    timestamp: float   # seconds
    destination: int

class SizeDistribution:
    def __init__(self, csv_file: str):
        self.sizes, self.cdf = [], []
        with open(csv_file, newline='') as f:
            for parts in csv.reader(f):
                if not parts or parts[0].startswith('#') or len(parts) < 2:
                    continue
                try:
                    size = int(parts[0])
                    prob = float(parts[1])
                except ValueError:
                    continue
                self.sizes.append(size)
                self.cdf.append(prob)
        if not self.sizes:
            raise ValueError(f"Empty CDF file: {csv_file}")
        if self.cdf[-1] < 1.0:
            self.cdf[-1] = 1.0
        # compute mean size
        pmf, prev = [], 0.0
        for p in self.cdf:
            pmf.append(p - prev)
            prev = p
        self.mean_size = sum(s * p for s, p in zip(self.sizes, pmf))

    def sample(self) -> int:
        """Sample size by inverse CDF."""
        u = random.random()
        idx = bisect.bisect_left(self.cdf, u)
        return self.sizes[min(idx, len(self.sizes) - 1)]

class FlowGenerator:
    def __init__(self, mean_interarrival: float,
                 size_dist: SizeDistribution,
                 type_names: list[str]):
        self.mean_interarrival = mean_interarrival
        self.size_dist = size_dist
        self.type_names = type_names

    def generate_flows(self, num_flows: int) -> list[Flow]:
        flows = []
        current_time = 0.0
        for _ in range(num_flows):
            # inter-arrival in seconds
            current_time += random.expovariate(1.0 / self.mean_interarrival)
            app = random.choice(self.type_names)
            if app == "type A":
                size = self.size_dist.sample()
                raw_rate = PARETO_SCALE * random.paretovariate(PARETO_SHAPE) * 4.0
                dest = random.randint(*DEST_RANGE_A)
            else:
                size = int(self.size_dist.mean_size)
                raw_rate = PARETO_SCALE * random.paretovariate(PARETO_SHAPE) / 4.0
                dest = random.randint(*DEST_RANGE_B)
            rate = min(raw_rate, MAX_RATE)
            flows.append(Flow(current_time, size, rate, dest, app))
        return flows

# Plot functions save PNGs
if HAS_PLOTTING:
    def plot_rate_distribution():
        raw = PARETO_SCALE * (np.random.pareto(PARETO_SHAPE, SAMPLE_COUNT) + 1)
        samples = np.minimum(raw, MAX_RATE)
        emp_mean = samples.mean()
        bins = np.logspace(np.log10(PARETO_SCALE), np.log10(MAX_RATE), BIN_COUNT)
        plt.figure()
        plt.hist(samples, bins=bins, density=True, alpha=0.6, label='Emp PDF')
        plt.xscale('log'); plt.yscale('log')
        x = np.logspace(np.log10(PARETO_SCALE), np.log10(MAX_RATE), 200)
        pdf = PARETO_SHAPE * (PARETO_SCALE**PARETO_SHAPE) / (x**(PARETO_SHAPE+1))
        plt.plot(x, pdf, 'k-', lw=2, label='PDF')
        tail = (PARETO_SCALE / MAX_RATE)**PARETO_SHAPE
        plt.scatter([MAX_RATE], [tail], color='k', s=50, label='Mass@MAX')
        plt.axvline(emp_mean, color='r', ls='--', label=f'Mean {emp_mean:.2e}s')
        plt.xlabel('Rate (bps)'); plt.ylabel('PDF')
        plt.legend(); plt.grid(True, which='both', ls='--', lw=0.5)
        fn = os.path.join(PLOT_DIR, 'rate_hist_trunc.png')
        plt.savefig(fn, bbox_inches='tight'); plt.close()
        print(f"[*] Saved: {fn}")
    def plot_rate_pdf():
        x = np.logspace(np.log10(PARETO_SCALE), np.log10(MAX_RATE), 300)
        pdf = PARETO_SHAPE * (PARETO_SCALE**PARETO_SHAPE) / (x**(PARETO_SHAPE+1))
        tail = (PARETO_SCALE / MAX_RATE)**PARETO_SHAPE
        plt.figure()
        plt.plot(x, pdf, label='PDF')
        plt.scatter([MAX_RATE], [tail], color='k', s=50, label='Mass@MAX')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Rate (bps)'); plt.ylabel('PDF')
        plt.legend(); plt.grid(True, which='both', ls='--', lw=0.5)
        fn = os.path.join(PLOT_DIR, 'rate_pdf_trunc.png')
        plt.savefig(fn, bbox_inches='tight'); plt.close()
        print(f"[*] Saved: {fn}")
else:
    def plot_rate_distribution(*args, **kwargs): print("[*] Plot disabled.")
    def plot_rate_pdf(*args, **kwargs): print("[*] Plot disabled.")


def main():
    # 1) Generate flows and save (times in seconds)
    size_dist = SizeDistribution(SIZE_CDF_CSV)
    gen = FlowGenerator(MEAN_INTERARRIVAL, size_dist, TYPE_NAMES)
    flows = gen.generate_flows(NUM_FLOWS)
    with open(FLOW_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start_time_s', 'size_bytes', 'rate_bps', 'dest', 'type'])
        for fl in flows:
            w.writerow([f"{fl.start_time:.6f}", fl.size, f"{fl.rate:.2f}", fl.destination, fl.type])
    print(f"[*] Saved flows to {FLOW_CSV}")

    # 2) Expand flows into packets and save
    packets = []
    for fl in flows:
        n = math.ceil(fl.size / PACKET_BYTES)
        interval = (PACKET_BYTES * 8.0) / fl.rate  # seconds
        t = fl.start_time
        for _ in range(n):
            t += random.expovariate(1.0 / interval)
            packets.append(Packet(timestamp=t, destination=fl.destination))
    packets.sort(key=lambda p: p.timestamp)
    with open(PACKET_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_s', 'dest'])
        for p in packets:
            w.writerow([f"{p.timestamp:.6f}", p.destination])
    print(f"[*] Saved packets to {PACKET_CSV}")

    # 3) Plots
    plot_rate_distribution()
    plot_rate_pdf()

if __name__ == '__main__':
    main()

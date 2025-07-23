#!/usr/bin/env python3
"""
traffic_generator.py

Generates traffic flows for multiple Top-of-Rack (ToR) switches,
then expands each flow into timestamped packets. Organizes output under
separate trace folders for easy management of multiple runs.

Outputs under TRACE_DIR (default "traces/trace_1"):
- traffic.csv: metadata of this trace (parameters, NUM_OF_TORS, etc.) plus flow list
- trace.csv: packet streams across all ToRs in a sorted list
- plots/*.png: rate distribution visualizations
All timestamps in seconds.
"""

import random
import bisect
import csv
import os
import math
from dataclasses import dataclass

# === Configuration ===
TRACE_NAME = "trace_1"                # Subfolder name for this run
BASE_DIR = "traces"                   # Parent folder for all traces
TRACE_DIR = os.path.join(BASE_DIR, TRACE_NAME)

# Pareto distribution parameters
PARETO_SHAPE = 1.5                     # Shape α (>1)
PARETO_MEAN = 2.5e9                    # Base mean rate (bps)
MAX_RATE = 40e9                        # Truncation threshold (bps)

# Aggregate incoming link
TARGET_BITRATE = 1.6e12                # 1.6 Tbps (bps)

# Applications and racks
TYPE_NAMES = ["type A", "type B"]    # 50:50 mix
NUM_OF_TORS = 10                        # Number of ToR switches

# Destination ID ranges
DEST_RANGE_A = (2000, 100_000_000)
DEST_RANGE_B = (1, 2000)

# Mean per-flow rates (bps)
MEAN_RATE_A = PARETO_MEAN * 4.0         # Faster flows
MEAN_RATE_B = PARETO_MEAN / 4.0         # Slower flows
AVG_FLOW_RATE = 0.5 * (MEAN_RATE_A + MEAN_RATE_B)

# Mean inter-arrival between flows (seconds)
MEAN_INTERARRIVAL = AVG_FLOW_RATE / TARGET_BITRATE

# Flow & packet settings
SIZE_CDF_CSV = "SizeDistributions/FB_Hadoop_Inter_Rack_FlowCDF.csv"
NUM_FLOWS = 10                          # Flows per ToR
PACKET_BYTES = 1500                     # Payload per packet (bytes)

# Plotting parameters
SAMPLE_COUNT = 1_000_000
BIN_COUNT = 100

# Create directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(TRACE_DIR, exist_ok=True)
PLOT_DIR = os.path.join(TRACE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
TRAFFIC_CSV = os.path.join(TRACE_DIR, "traffic.csv")
TRACE_CSV = os.path.join(TRACE_DIR, "trace.csv")

# Helper: compute truncated Pareto scale x_m
def compute_truncated_scale(alpha, desired_mean, M):
    low, high = 1e-9, M
    for _ in range(100):
        mid = 0.5 * (low + high)
        p_tail = (mid / M) ** alpha
        E_cont = alpha * mid / (alpha - 1) * (1 - (mid / M) ** (alpha - 1))
        if E_cont + M * p_tail > desired_mean:
            high = mid
        else:
            low = mid
    return mid

PARETO_SCALE = compute_truncated_scale(PARETO_SHAPE, PARETO_MEAN, MAX_RATE)
print(f"[*] Pareto scale x_m={PARETO_SCALE:.2e}, MAX_RATE={MAX_RATE:.2e}")

# Plotting libs
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("[*] Plotting disabled (numpy/matplotlib missing)")

# Data models
@dataclass
class Flow:
    tor_id: int
    start_time: float
    size: int
    rate: float
    destination: int
    type: str

@dataclass
class Packet:
    timestamp: float
    destination: int

# Size distribution from CDF
class SizeDistribution:
    def __init__(self, csv_file):
        self.sizes, self.cdf = [], []
        with open(csv_file, newline='') as f:
            for parts in csv.reader(f):
                if not parts or parts[0].startswith('#') or len(parts) < 2:
                    continue
                try:
                    s, p = int(parts[0]), float(parts[1])
                except ValueError:
                    continue
                self.sizes.append(s)
                self.cdf.append(p)
        if not self.sizes:
            raise ValueError(f"Empty CDF: {csv_file}")
        if self.cdf[-1] < 1.0:
            self.cdf[-1] = 1.0
        pmf, prev = [], 0.0
        for p in self.cdf:
            pmf.append(p - prev)
            prev = p
        self.mean_size = sum(s * prob for s, prob in zip(self.sizes, pmf))

    def sample(self) -> int:
        u = random.random()
        idx = bisect.bisect_left(self.cdf, u)
        return self.sizes[min(idx, len(self.sizes) - 1)]

# Flow generator
class FlowGenerator:
    def __init__(self, mi, size_dist, types):
        self.mi = mi
        self.sd = size_dist
        self.types = types

    def generate_flows(self, tor_id):
        flows, t = [], 0.0
        for _ in range(NUM_FLOWS):
            t += random.expovariate(1.0 / self.mi)
            app = random.choice(self.types)
            if app == "type A":
                size = self.sd.sample()
                raw = PARETO_SCALE * random.paretovariate(PARETO_SHAPE) * 4.0
                dest = random.randint(*DEST_RANGE_A)
            else:
                size = int(self.sd.mean_size)
                raw = PARETO_SCALE * random.paretovariate(PARETO_SHAPE) / 4.0
                dest = random.randint(*DEST_RANGE_B)
            rate = min(raw, MAX_RATE)
            flows.append(Flow(tor_id, t, size, rate, dest, app))
        return flows

# Plot functions
if HAS_PLOTTING:
    def plot_rate_histogram():
        raw = PARETO_SCALE * (np.random.pareto(PARETO_SHAPE, SAMPLE_COUNT) + 1)
        samp = np.minimum(raw, MAX_RATE)
        m = samp.mean()
        bins = np.logspace(np.log10(PARETO_SCALE), np.log10(MAX_RATE), BIN_COUNT)
        plt.figure()
        plt.hist(samp, bins=bins, density=True, alpha=0.6, label='Empirical')
        plt.xscale('log'); plt.yscale('log')
        x = np.logspace(np.log10(PARETO_SCALE), np.log10(MAX_RATE), 200)
        pdf = PARETO_SHAPE * (PARETO_SCALE**PARETO_SHAPE) / (x**(PARETO_SHAPE + 1))
        plt.plot(x, pdf, 'k-', lw=2, label='Theory')
        tail = (PARETO_SCALE / MAX_RATE)**PARETO_SHAPE
        plt.scatter([MAX_RATE], [tail], color='k', s=50, label='Mass@MAX')
        plt.axvline(m, color='r', ls='--', label=f'Mean={m:.2e}')
        plt.xlabel('Rate (bps)'); plt.ylabel('PDF'); plt.legend(); plt.grid(True, which='both', ls='--', lw=0.5)
        fn = os.path.join(PLOT_DIR, 'rate_hist.png'); plt.savefig(fn, bbox_inches='tight'); plt.close(); print(f"[*] Saved {fn}")

    def plot_rate_pdf():
        x = np.logspace(np.log10(PARETO_SCALE), np.log10(MAX_RATE), 300)
        pdf = PARETO_SHAPE * (PARETO_SCALE**PARETO_SHAPE) / (x**(PARETO_SHAPE + 1))
        tail = (PARETO_SCALE / MAX_RATE)**PARETO_SHAPE
        plt.figure()
        plt.plot(x, pdf, label='Theory')
        plt.scatter([MAX_RATE], [tail], color='k', s=50, label='Mass@MAX')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Rate (bps)'); plt.ylabel('PDF'); plt.legend(); plt.grid(True, which='both', ls='--', lw=0.5)
        fn = os.path.join(PLOT_DIR, 'rate_pdf.png'); plt.savefig(fn, bbox_inches='tight'); plt.close(); print(f"[*] Saved {fn}")
else:
    def plot_rate_histogram(*a,**k): print("[*] Plot disabled")
    def plot_rate_pdf(*a,**k): print("[*] Plot disabled")

# Main execution
size_dist = SizeDistribution(SIZE_CDF_CSV)
gen = FlowGenerator(MEAN_INTERARRIVAL, size_dist, TYPE_NAMES)
tor_flows = {tor: gen.generate_flows(tor) for tor in range(1, NUM_OF_TORS + 1)}
all_flows = [fl for fls in tor_flows.values() for fl in fls]

# Write traffic.csv with metadata and flows
def write_traffic():
    with open(TRAFFIC_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value'])
        meta = [
            ('TRACE_NAME', TRACE_NAME),
            ('NUM_OF_TORS', NUM_OF_TORS),
            ('NUM_FLOWS', NUM_FLOWS),
            ('PACKET_BYTES', PACKET_BYTES),
            ('PARETO_SHAPE', PARETO_SHAPE),
            ('PARETO_MEAN', PARETO_MEAN),
            ('MAX_RATE', MAX_RATE),
            ('TARGET_BITRATE', TARGET_BITRATE),
            ('MEAN_INTERARRIVAL', MEAN_INTERARRIVAL),
            ('APP_MIX', '0.5-0.5'),
            ('APP_B_RANGE', f"[{DEST_RANGE_B[0]}, {DEST_RANGE_B[1]}]"),
            ('SIZE_DISTRIBUTION', os.path.basename(SIZE_CDF_CSV)),
        ]
        for param in meta:
            w.writerow(param)
        w.writerow([])
        w.writerow(['tor_id', 'start_time_s', 'size_bytes', 'rate_bps', 'destination', 'type'])
        for fl in all_flows:
            w.writerow([fl.tor_id, f"{fl.start_time:.6f}", fl.size, f"{fl.rate:.2f}", fl.destination, fl.type])
    print(f"[*] Saved traffic data to {TRAFFIC_CSV}")

# Write trace.csv as a single sorted list of all packets across ToRs
def write_trace():
    # Collect all packets from all ToRs
    all_packets = []  # list of (timestamp, destination, tor_id)
    for tor in range(1, NUM_OF_TORS+1):
        for fl in tor_flows[tor]:
            num_pkts = math.ceil(fl.size / PACKET_BYTES)
            interval = (PACKET_BYTES * 8.0) / fl.rate
            t = fl.start_time
            for _ in range(num_pkts):
                t += random.expovariate(1.0 / interval)
                all_packets.append((t, fl.destination, tor))
    # Sort all packets by timestamp
    all_packets.sort(key=lambda x: x[0])
    # Write to CSV
    with open(TRACE_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_s', 'destination', 'tor_id'])
        for ts, dest, tor in all_packets:
            w.writerow([f"{ts:.6f}", dest, tor])
    print(f"[*] Saved packet trace to {TRACE_CSV}")

# Execute all
def main():
    write_traffic()
    write_trace()
    if HAS_PLOTTING:
        plot_rate_histogram()
        plot_rate_pdf()

if __name__ == '__main__':
    main()

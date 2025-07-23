#!/usr/bin/env python3
"""
traffic_generator.py

Generates traffic flows for multiple Top-of-Rack (ToR) switches,
then expands each flow into timestamped packets. Outputs:
- flows.csv: all flows across ToRs with tor_id
- packets_tor_<id>.csv: per-ToR sorted packet streams
- plots/*.png: rate distribution visualizations
"""

import random
import bisect
import csv
import os
import math
from dataclasses import dataclass

# === Configuration ===
# Pareto rate distribution
PARETO_SHAPE = 1.5
PARETO_MEAN = 2.5e9
MAX_RATE = 40e9

# Aggregate link target
TARGET_BITRATE = 1.6e12  # 1.6 Tbps

# Apps and ToRs
TYPE_NAMES = ["type A", "type B"]
NUM_OF_TORS = 10

# Per-type destinations
DEST_RANGE_A = (2000, 100_000_000)
DEST_RANGE_B = (1, 2000)

# Mean flow rates per type
MEAN_RATE_A = PARETO_MEAN * 4.0
MEAN_RATE_B = PARETO_MEAN / 4.0
AVG_FLOW_RATE = 0.5 * (MEAN_RATE_A + MEAN_RATE_B)

# Flow inter-arrival to hit TARGET
MEAN_INTERARRIVAL = AVG_FLOW_RATE / TARGET_BITRATE  # seconds

# Flow & packet params
SIZE_CDF_CSV = "SizeDistributions/FB_Hadoop_Inter_Rack_FlowCDF.csv"
NUM_FLOWS = 10
PACKET_BYTES = 1500

# Output dirs
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FLOW_CSV = os.path.join(OUTPUT_DIR, "flows.csv")
PACKET_CSV_TEMPLATE = os.path.join(OUTPUT_DIR, "packets_tor_{tor}.csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Compute truncated Pareto scale x_m
def compute_truncated_scale(alpha, desired_mean, M):
    low, high = 1e-9, M
    for _ in range(100):
        mid = 0.5*(low+high)
        p_tail = (mid/M)**alpha
        E_cont = alpha*mid/(alpha-1)*(1-(mid/M)**(alpha-1))
        if E_cont + M*p_tail > desired_mean:
            high = mid
        else:
            low = mid
    return mid

PARETO_SCALE = compute_truncated_scale(PARETO_SHAPE, PARETO_MEAN, MAX_RATE)
# Sampling config for plotting
SAMPLE_COUNT = 1000000  # Number of samples for empirical PDF
BIN_COUNT = 100         # Number of bins in histogram
print(f"[*] Pareto x_m={PARETO_SCALE:.2e}, MAX_RATE={MAX_RATE:.2e}")

# Plotting dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("[*] Plotting disabled (numpy/matplotlib missing)")

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

class SizeDistribution:
    def __init__(self, csv_file):
        self.sizes, self.cdf = [], []
        with open(csv_file, newline='') as f:
            for parts in csv.reader(f):
                if not parts or parts[0].startswith('#') or len(parts)<2: continue
                try:
                    s=int(parts[0]); p=float(parts[1])
                except: continue
                self.sizes.append(s); self.cdf.append(p)
        if not self.sizes: raise ValueError("Empty CDF: %s"%csv_file)
        if self.cdf[-1]<1.0: self.cdf[-1]=1.0
        prev=0.0; pmf=[]
        for p in self.cdf: pmf.append(p-prev); prev=p
        self.mean_size = sum(s*prob for s,prob in zip(self.sizes,pmf))
    def sample(self):
        u=random.random(); idx=bisect.bisect_left(self.cdf,u)
        return self.sizes[min(idx,len(self.sizes)-1)]

class FlowGenerator:
    def __init__(self, mean_interarrival, size_dist, type_names):
        self.mi = mean_interarrival; self.sd=size_dist; self.types=type_names
    def generate_flows(self, tor):
        fls=[]; t=0.0
        for _ in range(NUM_FLOWS):
            t += random.expovariate(1.0/self.mi)
            app=random.choice(self.types)
            if app=="type A": size=self.sd.sample(); raw=PARETO_SCALE*random.paretovariate(PARETO_SHAPE)*4.0; dest=random.randint(*DEST_RANGE_A)
            else: size=int(self.sd.mean_size); raw=PARETO_SCALE*random.paretovariate(PARETO_SHAPE)/4.0; dest=random.randint(*DEST_RANGE_B)
            rate=min(raw,MAX_RATE)
            fls.append(Flow(tor,t,size,rate,dest,app))
        return fls

# Plot functions
if HAS_PLOTTING:
    def plot_rate():
        raw=PARETO_SCALE*(np.random.pareto(PARETO_SHAPE,SAMPLE_COUNT)+1)
        s=np.minimum(raw,MAX_RATE); m=s.mean()
        bins=np.logspace(np.log10(PARETO_SCALE),np.log10(MAX_RATE),BIN_COUNT)
        plt.figure(); plt.hist(s,bins=bins,density=True,alpha=0.6,label='Emp'); plt.xscale('log'); plt.yscale('log')
        x=np.logspace(np.log10(PARETO_SCALE),np.log10(MAX_RATE),200)
        pdf=PARETO_SHAPE*(PARETO_SCALE**PARETO_SHAPE)/(x**(PARETO_SHAPE+1))
        plt.plot(x,pdf,'k-',lw=2,label='PDF')
        pt=(PARETO_SCALE/MAX_RATE)**PARETO_SHAPE
        plt.scatter([MAX_RATE],[pt],color='k',s=50,label='MASS')
        plt.axvline(m,color='r',ls='--',label=f'M={m:.2e}');plt.legend();plt.grid(True,which='both',ls='--',lw=0.5)
        fn=os.path.join(PLOT_DIR,'rate_hist.png');plt.savefig(fn,bbox_inches='tight');plt.close();print(f"[*] Saved {fn}")
    def plot_pdf():
        x=np.logspace(np.log10(PARETO_SCALE),np.log10(MAX_RATE),300)
        pdf=PARETO_SHAPE*(PARETO_SCALE**PARETO_SHAPE)/(x**(PARETO_SHAPE+1))
        pt=(PARETO_SCALE/MAX_RATE)**PARETO_SHAPE
        plt.figure();plt.plot(x,pdf,label='PDF');plt.scatter([MAX_RATE],[pt],color='k',s=50,label='MASS')
        plt.xscale('log');plt.yscale('log');plt.legend();plt.grid(True,which='both',ls='--',lw=0.5)
        fn=os.path.join(PLOT_DIR,'rate_pdf.png');plt.savefig(fn,bbox_inches='tight');plt.close();print(f"[*] Saved {fn}")
else:
    def plot_rate(*a,**k):print("[*] Plot disabled")
    def plot_pdf(*a,**k):print("[*] Plot disabled")

# Main
size_dist=SizeDistribution(SIZE_CDF_CSV)
gen=FlowGenerator(MEAN_INTERARRIVAL,size_dist,TYPE_NAMES)
all_flows=[]
for tor in range(1,NUM_OF_TORS+1):
    all_flows+=gen.generate_flows(tor)
# Save flows
with open(FLOW_CSV,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['tor_id','start_time_s','size_b','rate_bps','dest','type'])
    for fl in all_flows:
        w.writerow([fl.tor_id,f"{fl.start_time:.6f}",fl.size,f"{fl.rate:.2f}",fl.destination,fl.type])
print(f"[*] Saved flows to {FLOW_CSV}")
# Per-TOR packets
for tor in range(1,NUM_OF_TORS+1):
    flows=[f for f in all_flows if f.tor_id==tor]
    pkts=[]
    for fl in flows:
        n=math.ceil(fl.size/PACKET_BYTES)
        interval=(PACKET_BYTES*8.0)/fl.rate
        t=fl.start_time
        for _ in range(n): t+=random.expovariate(1.0/interval); pkts.append(Packet(t,fl.destination))
    pkts.sort(key=lambda p:p.timestamp)
    fn=PACKET_CSV_TEMPLATE.format(tor=tor)
    with open(fn,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['timestamp_s','dest'])
        for p in pkts: w.writerow([f"{p.timestamp:.6f}",p.destination])
    print(f"[*] Saved packets to {fn}")
# Plots\ if HAS_PLOTTING:
    plot_rate(); plot_pdf()

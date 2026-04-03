#!/usr/bin/env python3
# parse_times.py — ECE569 HW4
# Parses experiment output files, prints timing tables,
# generates plots as PDF, and injects values into hw4_report.tex
#
# Usage (run from ece569/labs/hw4/report/):
#   python3 parse_times.py                          # table + graphs only
#   python3 parse_times.py --inject hw4_report.tex  # also injects into LaTeX
#
# Outputs written to ece569/build_dir/Histogram_output/figures/:
#   fig_exp1_runs.pdf       — run-by-run line chart, Experiment 1
#   fig_exp2_runs.pdf       — run-by-run line chart, Experiment 2
#   fig_exp1_bar.pdf        — average bar chart with std dev, Experiment 1
#   fig_exp2_bar.pdf        — average bar chart with std dev, Experiment 2
#   fig_avg_comparison.pdf  — grouped bar: E1 vs E2 per version side-by-side
#   fig_exp1_speedup.pdf    — speedup over V0 baseline, Experiment 1
#   fig_exp2_speedup.pdf    — speedup over V0 baseline, Experiment 2

import re, sys, pathlib, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this script's location so it works from any cwd
SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
OUTPUT_ROOT = (SCRIPT_DIR / "../../../build_dir/Histogram_output").resolve()
EXP1_DIR    = OUTPUT_ROOT / "exp1"
EXP2_DIR    = OUTPUT_ROOT / "exp2"
FIG_DIR     = OUTPUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

VERSION_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#3BB273", "#C0392B", "#8E44AD", "#1ABC9C", "#E67E22"]
VERSION_LABELS = [
    "V0 \u2014 Global Scatter",
    "V1 \u2014 Block Privatization",
    "V2 \u2014 R=2 + RLE (GOAT)",
    "V3 \u2014 R=2 Ablation (no RLE)",
    "V4 \u2014 Warp Aggregation",
    "V5 \u2014 Bin-Centric Gather",
    "V6 \u2014 Sort & Reduce-by-Key",
    "V7 \u2014 Multi-split",
]
VERSION_MARKS  = ["o", "s", "^", "D", "v", "P", "X", "*"]
RUNS           = list(range(1, 11))
NUM_VERSIONS   = 8   # V0 through V7

MUTED_VERSIONS = {5}  # V5 Bin-Centric Gather — greyed out in all-version plots

def _vc(v):
    """Per-version color: grey for muted versions, normal otherwise."""
    return "#CCCCCC" if v in MUTED_VERSIONS else VERSION_COLORS[v]

# ── bandwidth calculation ─────────────────────────────────────────────────────
def calculate_bandwidth(size_n, time_ms):
    """Effective Memory Bandwidth in GB/s: N elements * 4 B / time."""
    if np.isnan(time_ms) or time_ms <= 0:
        return float("nan")
    return (size_n * 4) / (time_ms / 1000.0) / 1e9

# ── time extraction ───────────────────────────────────────────────────────────
# Matches the actual output line from solution.cu:
#   "Total compute time (ms) X.XXXXXX for version N"
TIME_PATTERNS = [
    r"Total compute time \(ms\)\s+([0-9]+\.?[0-9]*)",   # primary: matches actual output
    r"(?i)kernel\s+time[:\s]+([0-9]+\.?[0-9]*)",
    r"(?i)elapsed[:\s]+([0-9]+\.?[0-9]*)",
    r"(?i)time[:\s]+([0-9]+\.?[0-9]*)\s*ms",
    r"(?i)([0-9]+\.[0-9]+)\s*ms",
]

def extract_time(filepath):
    try:
        text = filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for pat in TIME_PATTERNS:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None

def collect(exp_dir, version, n_runs=10):
    times = []
    for run in range(1, n_runs + 1):
        f = exp_dir / f"v{version}_run{run}.txt"
        t = extract_time(f)
        if t is None:
            print(f"  WARNING: could not parse time from {f}")
        times.append(t)
    return times

def valid(times):     return [t for t in times if t is not None]
def vavg(times):      v = valid(times); return statistics.mean(v)  if v            else float("nan")
def vstdev(times):    v = valid(times); return statistics.stdev(v) if len(v) > 1   else 0.0
def fmt(t):           return f"{t:.4f}" if t is not None else "N/A"
def nan_list(times):  return [t if t is not None else float("nan") for t in times]

# ── collect all data ──────────────────────────────────────────────────────────
print("Parsing experiment outputs...\n")
print(f"Reading from: {OUTPUT_ROOT}\n")
e1 = [collect(EXP1_DIR, v) for v in range(NUM_VERSIONS)]
e2 = [collect(EXP2_DIR, v) for v in range(NUM_VERSIONS)]

# ── print tables ──────────────────────────────────────────────────────────────
def print_table(label, data, size_n=500000):
    print(f"{'='*115}")
    print(f"  {label}")
    print(f"{'='*115}")
    print(f"  {'Run':<5} {'V0 (ms)':<14} {'V1 (ms)':<14} {'V2 (ms)':<14} {'V3 (ms)':<14} {'V4 (ms)':<14} {'V5 (ms)':<14} {'V6 (ms)':<14} {'V7 (ms)':<14}")
    print(f"  {'-'*110}")
    for i in range(10):
        row = [fmt(data[v][i]) if i < len(data[v]) else "N/A" for v in range(NUM_VERSIONS)]
        print(f"  {i+1:<5} {row[0]:<14} {row[1]:<14} {row[2]:<14} {row[3]:<14} {row[4]:<14} {row[5]:<14} {row[6]:<14} {row[7]:<14}")
    print(f"  {'-'*110}")
    avgs = [vavg(data[v]) for v in range(NUM_VERSIONS)]
    avg_strs = [f"{a:.4f}" if not np.isnan(a) else "N/A" for a in avgs]
    print(f"  {'Avg':<5} {avg_strs[0]:<14} {avg_strs[1]:<14} {avg_strs[2]:<14} {avg_strs[3]:<14} {avg_strs[4]:<14} {avg_strs[5]:<14} {avg_strs[6]:<14} {avg_strs[7]:<14}")
    bws = [calculate_bandwidth(size_n, a) for a in avgs]
    bw_strs = [f"{b:.2f} GB/s" if not np.isnan(b) else "N/A" for b in bws]
    print(f"  {'BW':<5} {bw_strs[0]:<14} {bw_strs[1]:<14} {bw_strs[2]:<14} {bw_strs[3]:<14} {bw_strs[4]:<14} {bw_strs[5]:<14} {bw_strs[6]:<14} {bw_strs[7]:<14}")
    print()

print_table("Experiment 1 — Random data (dataset 6, 500k elements)", e1)
print_table("Experiment 2 — Uniform data (500k same-value elements)", e2)

# ════════════════════════════════════════════════════════════
#  FIG 1 & 2 — Run-by-run line charts
# ════════════════════════════════════════════════════════════
def plot_runs(data, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in range(NUM_VERSIONS):
        if v in MUTED_VERSIONS:
            continue
        ax.plot(RUNS, nan_list(data[v]),
                marker=VERSION_MARKS[v],
                color=VERSION_COLORS[v],
                label=VERSION_LABELS[v],
                linewidth=1.6,
                markersize=6)
    ax.set_xlabel("Run number")
    ax.set_ylabel("Kernel execution time (ms)")
    ax.set_title(title)
    ax.set_xticks(RUNS)
    ax.margins(y=0.15)
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

plot_runs(e1, "Experiment 1 — Kernel time per run (random data)",  "fig_exp1_runs.pdf")
plot_runs(e2, "Experiment 2 — Kernel time per run (uniform data)", "fig_exp2_runs.pdf")

# ════════════════════════════════════════════════════════════
#  FIG 1b & 2b — Run-by-run line charts, LOG-SCALE (all versions)
#  Logarithmic y-axis lets V3 (~11 ms) and V0-V2 (~0.18 ms)
#  coexist on the same plot without one dominating the scale.
# ════════════════════════════════════════════════════════════
def plot_runs_log(data, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in range(NUM_VERSIONS):
        if v in MUTED_VERSIONS:
            continue
        ax.plot(RUNS, nan_list(data[v]),
                marker=VERSION_MARKS[v],
                color=VERSION_COLORS[v],
                label=VERSION_LABELS[v],
                linewidth=1.6,
                markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel("Run number")
    ax.set_ylabel("Kernel execution time (ms, log scale)")
    ax.set_title(title)
    ax.set_xticks(RUNS)
    ax.margins(y=0.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

plot_runs_log(e1, "Experiment 1 — Kernel time per run, log scale (all versions, random)",  "fig_exp1_runs_log.pdf")
plot_runs_log(e2, "Experiment 2 — Kernel time per run, log scale (all versions, uniform)", "fig_exp2_runs_log.pdf")

# ════════════════════════════════════════════════════════════
#  FIG 3 & 4 — Average bar charts with std dev error bars
# ════════════════════════════════════════════════════════════
XLABELS = [
    "V0\nGlobal\nScatter",
    "V1\nBlock\nPrivatization",
    "V2\nR=2+RLE\n(GOAT)",
    "V3\nR=2\nAblation",
    "V4\nWarp\nAggregation",
    "V5\nBin-Centric\nGather",
    "V6\nSort &\nReduce",
    "V7\nMulti-\nsplit",
]

def plot_avg_bar(data, title, filename):
    vlist  = [v for v in range(NUM_VERSIONS) if v not in MUTED_VERSIONS]
    avgs   = [vavg(data[v])     for v in vlist]
    errs   = [vstdev(data[v])   for v in vlist]
    xlbls  = [XLABELS[v]        for v in vlist]
    colors = [VERSION_COLORS[v] for v in vlist]
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(len(vlist))
    bars = ax.bar(x, avgs, yerr=errs,
                  color=colors, width=0.5,
                  capsize=5, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title(title)
    err_max = max((e for e in errs if not np.isnan(e)), default=0)
    for bar, val in zip(bars, avgs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err_max * 0.08 + 0.001,
                    f"{val:.3f} ms", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

plot_avg_bar(e1, "Experiment 1 — Average kernel time (random data)",  "fig_exp1_bar.pdf")
plot_avg_bar(e2, "Experiment 2 — Average kernel time (uniform data)", "fig_exp2_bar.pdf")

# ════════════════════════════════════════════════════════════
#  FIG 5 — Grouped bar: E1 vs E2 side-by-side per version
# ════════════════════════════════════════════════════════════
_bvlist = [v for v in range(NUM_VERSIONS) if v not in MUTED_VERSIONS]
_bxlbls = [XLABELS[v]        for v in _bvlist]
_bclrs  = [VERSION_COLORS[v] for v in _bvlist]
fig, ax = plt.subplots(figsize=(8, 4.5))
x      = np.arange(len(_bvlist))
width  = 0.32
avgs1  = [vavg(e1[v])   for v in _bvlist]
avgs2  = [vavg(e2[v])   for v in _bvlist]
errs1  = [vstdev(e1[v]) for v in _bvlist]
errs2  = [vstdev(e2[v]) for v in _bvlist]

ax.bar(x - width/2, avgs1, width, yerr=errs1,
       color=_bclrs, alpha=0.90,
       capsize=4, error_kw={"linewidth": 1.1})
ax.bar(x + width/2, avgs2, width, yerr=errs2,
       color=_bclrs, alpha=0.45,
       capsize=4, error_kw={"linewidth": 1.1}, hatch="//")

ax.legend(handles=[
    Patch(facecolor="#888", alpha=0.90, label="Experiment 1 — random"),
    Patch(facecolor="#888", alpha=0.45, hatch="//", label="Experiment 2 — uniform"),
], loc="upper right")
ax.set_xticks(x)
ax.set_xticklabels(_bxlbls)
ax.set_ylabel("Average kernel time (ms)")
ax.set_title("Experiment 1 vs. 2 — Average kernel time by version")
fig.tight_layout()
out = FIG_DIR / "fig_avg_comparison.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ════════════════════════════════════════════════════════════
#  FIG 6 & 7 — Speedup over V0 baseline
# ════════════════════════════════════════════════════════════
def plot_speedup(data, title, filename, outside_legend=False):
    vlist        = [v for v in range(NUM_VERSIONS) if v not in MUTED_VERSIONS]
    baseline     = vavg(data[0])
    baseline_std = vstdev(data[0])
    if np.isnan(baseline) or baseline == 0:
        print(f"  SKIP speedup chart (no V0 baseline): {filename}")
        return
    speedups = []
    s_errs   = []
    for v in vlist:
        avg_v = vavg(data[v])
        std_v = vstdev(data[v])
        if not np.isnan(avg_v) and avg_v > 0:
            sp = baseline / avg_v
            rel = np.sqrt((baseline_std / baseline) ** 2 + (std_v / avg_v) ** 2)
            speedups.append(sp)
            s_errs.append(sp * rel)
        else:
            speedups.append(float("nan"))
            s_errs.append(0.0)
    xlbls  = [XLABELS[v]        for v in vlist]
    colors = [VERSION_COLORS[v] for v in vlist]
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(len(vlist))
    bars = ax.bar(x, speedups, yerr=s_errs, color=colors, width=0.5,
                  capsize=5, error_kw={"linewidth": 1.2})
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5, label="V0 baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel("Speedup relative to V0")
    ax.set_title(title)
    if outside_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=9)
    else:
        ax.legend(loc="lower right")
    for bar, val in zip(bars, speedups):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}×", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

plot_speedup(e1, "Experiment 1 — Speedup relative to V0 (random data)",  "fig_exp1_speedup.pdf")
plot_speedup(e2, "Experiment 2 — Speedup relative to V0 (uniform data)", "fig_exp2_speedup.pdf", outside_legend=True)

# ════════════════════════════════════════════════════════════
#  NO-SLOW variants: exclude V5 (Bin-Centric Gather, O(N*B)) and
#  optionally V6 (Thrust sort, includes O(N log N) sort overhead).
#  V2/V3/V4/V7 all have similar ms-range times to V0/V1.
# ════════════════════════════════════════════════════════════
NOV3_VERSIONS = [0, 1, 2, 3, 4, 6, 7]  # exclude V5 (Gather, ~O(N*B) — different timescale)
NOV3_COLORS   = [VERSION_COLORS[v] for v in NOV3_VERSIONS]
NOV3_LABELS   = [VERSION_LABELS[v] for v in NOV3_VERSIONS]
NOV3_MARKS    = [VERSION_MARKS[v]  for v in NOV3_VERSIONS]
NOV3_XLABELS  = [XLABELS[v]        for v in NOV3_VERSIONS]

def plot_runs_nov3(data, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, v in enumerate(NOV3_VERSIONS):
        ax.plot(RUNS, nan_list(data[v]),
                marker=NOV3_MARKS[idx],
                color=NOV3_COLORS[idx],
                label=NOV3_LABELS[idx],
                linewidth=1.6, markersize=6)
    ax.set_xlabel("Run number")
    ax.set_ylabel("Kernel execution time (ms)")
    ax.set_title(title)
    ax.set_xticks(RUNS)
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_avg_bar_nov3(data, title, filename):
    avgs = [vavg(data[v])   for v in NOV3_VERSIONS]
    errs = [vstdev(data[v]) for v in NOV3_VERSIONS]
    fig, ax = plt.subplots(figsize=(6, 4))
    x    = np.arange(len(NOV3_VERSIONS))
    bars = ax.bar(x, avgs, yerr=errs,
                  color=NOV3_COLORS, width=0.5,
                  capsize=5, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(NOV3_XLABELS)
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title(title)
    err_max = max((e for e in errs if not np.isnan(e)), default=0)
    for bar, val in zip(bars, avgs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err_max * 0.08 + 0.001,
                    f"{val:.3f} ms", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_speedup_nov3(data, title, filename):
    baseline = vavg(data[0])
    if np.isnan(baseline) or baseline == 0:
        print(f"  SKIP speedup chart (no V0 baseline): {filename}")
        return
    speedups = [baseline / vavg(data[v]) if not np.isnan(vavg(data[v])) and vavg(data[v]) > 0
                else float("nan") for v in NOV3_VERSIONS]
    fig, ax = plt.subplots(figsize=(6, 4))
    x    = np.arange(len(NOV3_VERSIONS))
    bars = ax.bar(x, speedups, color=NOV3_COLORS, width=0.5)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5, label="V0 baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(NOV3_XLABELS)
    ax.set_ylabel("Speedup relative to V0")
    ax.set_title(title)
    ax.legend(loc="upper left")
    for bar, val in zip(bars, speedups):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}\u00d7", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_comparison_nov3(e1_data, e2_data, title, filename):
    avgs1  = [vavg(e1_data[v])   for v in NOV3_VERSIONS]
    avgs2  = [vavg(e2_data[v])   for v in NOV3_VERSIONS]
    errs1  = [vstdev(e1_data[v]) for v in NOV3_VERSIONS]
    errs2  = [vstdev(e2_data[v]) for v in NOV3_VERSIONS]
    fig, ax = plt.subplots(figsize=(7, 4))
    x     = np.arange(len(NOV3_VERSIONS))
    width = 0.32
    ax.bar(x - width/2, avgs1, width, yerr=errs1,
           color=NOV3_COLORS, alpha=0.90,
           capsize=4, error_kw={"linewidth": 1.1})
    ax.bar(x + width/2, avgs2, width, yerr=errs2,
           color=NOV3_COLORS, alpha=0.45,
           capsize=4, error_kw={"linewidth": 1.1}, hatch="//")
    ax.legend(handles=[
        Patch(facecolor="#888", alpha=0.90, label="Experiment 1 \u2014 random"),
        Patch(facecolor="#888", alpha=0.45, hatch="//", label="Experiment 2 \u2014 uniform"),
    ], loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(NOV3_XLABELS)
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title(title)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print("\nGenerating without-V5 variants (excluding Bin-Centric Gather for readable y-axis scale)...")
plot_runs_nov3(e1, "Experiment 1 — Kernel time per run (V0–V2, random)",   "fig_exp1_runs_nov3.pdf")
plot_runs_nov3(e2, "Experiment 2 — Kernel time per run (V0–V2, uniform)",  "fig_exp2_runs_nov3.pdf")
plot_avg_bar_nov3(e1, "Experiment 1 — Average kernel time (V0–V2, random)",  "fig_exp1_bar_nov3.pdf")
plot_avg_bar_nov3(e2, "Experiment 2 — Average kernel time (V0–V2, uniform)", "fig_exp2_bar_nov3.pdf")
plot_speedup_nov3(e1, "Experiment 1 — Speedup V0–V2 (random)",  "fig_exp1_speedup_nov3.pdf")
plot_speedup_nov3(e2, "Experiment 2 — Speedup V0–V2 (uniform)", "fig_exp2_speedup_nov3.pdf")
plot_comparison_nov3(e1, e2, "Exp 1 vs. 2 — V0–V2 average kernel time", "fig_avg_comparison_nov3.pdf")

# ════════════════════════════════════════════════════════════
#  FIG: V2 vs V3 ablation — isolates RLE compression contribution
# ════════════════════════════════════════════════════════════
def plot_v2_v3_ablation(e1_data, e2_data, filename):
    """Bar chart: V3 (no RLE) vs V2 (with RLE) for Exp1 and Exp2.
    Difference isolates the pure RLE temporal compression speedup."""
    v2_e1 = vavg(e1_data[2]); v3_e1 = vavg(e1_data[3])
    v2_e2 = vavg(e2_data[2]); v3_e2 = vavg(e2_data[3])
    vals   = [v3_e1, v2_e1, v3_e2, v2_e2]
    colors = [VERSION_COLORS[3], VERSION_COLORS[2], VERSION_COLORS[3], VERSION_COLORS[2]]
    labels = ["V3 R=2, no RLE\n(Exp1 random)",  "V2 R=2+RLE\n(Exp1 random)",
              "V3 R=2, no RLE\n(Exp2 uniform)", "V2 R=2+RLE\n(Exp2 uniform)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(4)
    bars = ax.bar(x, vals, color=colors, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title("RLE Compression Impact: V3 (no RLE) vs V2 (R=2+RLE)")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f} ms", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

plot_v2_v3_ablation(e1, e2, "fig_v2_v3_ablation.pdf")

# ════════════════════════════════════════════════════════════
#  FIG: V5 Gather — O(N*B) arithmetic overhead visualization
# ════════════════════════════════════════════════════════════
def plot_v5_context(e1_data, e2_data, filename):
    """Bar chart: V0, V1, V5 side-by-side for Exp1 and Exp2.
    Shows that Bin-Centric Gather (V5) has far higher latency
    due to O(N*B) comparisons despite zero atomics."""
    versions   = [0, 1, 5]
    labels_ctx = ["V0 Global\nScatter", "V1 Block\nPrivatization", "V5 Bin-Centric\nGather"]
    colors_ctx = [VERSION_COLORS[v] for v in versions]
    avgs1 = [vavg(e1_data[v]) for v in versions]
    avgs2 = [vavg(e2_data[v]) for v in versions]
    errs1 = [vstdev(e1_data[v]) for v in versions]
    errs2 = [vstdev(e2_data[v]) for v in versions]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(versions))
    width = 0.32
    ax.bar(x - width/2, avgs1, width, yerr=errs1, color=colors_ctx, alpha=0.90,
           capsize=4, error_kw={"linewidth": 1.1})
    ax.bar(x + width/2, avgs2, width, yerr=errs2, color=colors_ctx, alpha=0.45,
           capsize=4, error_kw={"linewidth": 1.1}, hatch="//")
    ax.legend(handles=[
        Patch(facecolor="#888", alpha=0.90, label="Experiment 1 \u2014 random"),
        Patch(facecolor="#888", alpha=0.45, hatch="//", label="Experiment 2 \u2014 uniform"),
    ], loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ctx)
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title("O(N\u00d7B) Cost: V5 Bin-Centric Gather vs. V0/V1")
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

plot_v5_context(e1, e2, "fig_v5_gather_overhead.pdf")

# ════════════════════════════════════════════════════════════
#  SCALING EXPERIMENTS — V0, V1, V2 across input sizes
#  Reads from Histogram_output/scaling/v{V}_ds{DS}_run{R}.txt
#  Datasets: 6=500k, 8=1M, 9=10M, 10=50M (random, 4096 bins)
# ════════════════════════════════════════════════════════════
SCALING_DIR      = OUTPUT_ROOT / "scaling"
SCALING_DS_ORDER = [11, 12, 13, 14, 6, 8, 9, 10]
SCALING_DS_SIZES = {
    11:      2_000,
    12:     10_000,
    13:     50_000,
    14:    200_000,
    6:     500_000,
    8:   1_000_000,
    9:  10_000_000,
    10: 50_000_000,
}
SCALING_VERSIONS = list(range(NUM_VERSIONS))   # V0–V7
SCALING_RUNS     = 5

def collect_scaling():
    sc = {}
    for v in SCALING_VERSIONS:
        sc[v] = {}
        for ds in SCALING_DS_ORDER:
            times = []
            for run in range(1, SCALING_RUNS + 1):
                f = SCALING_DIR / f"v{v}_ds{ds}_run{run}.txt"
                t = extract_time(f)
                if t is None:
                    print(f"  WARNING: missing {f.name}")
                times.append(t)
            sc[v][ds] = times
    return sc

def _fmt_n(n):
    if n >= 1_000_000:
        return f"{int(n / 1_000_000)}M"
    return f"{int(n / 1_000)}k"

def plot_scaling_time(sc, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [SCALING_DS_SIZES[ds] for ds in SCALING_DS_ORDER]
    for v in SCALING_VERSIONS:
        ys = [vavg(sc[v][ds])   for ds in SCALING_DS_ORDER]
        es = [vstdev(sc[v][ds]) for ds in SCALING_DS_ORDER]
        mask = [not np.isnan(y) for y in ys]
        xp = [x for x, m in zip(xs, mask) if m]
        yp = [y for y, m in zip(ys, mask) if m]
        ep = [e for e, m in zip(es, mask) if m]
        if xp:
            ax.errorbar(xp, yp, yerr=ep,
                        marker=VERSION_MARKS[v], color=VERSION_COLORS[v],
                        label=VERSION_LABELS[v], linewidth=1.6, markersize=6, capsize=4)
    ax.set_xlabel("Input size $N$")
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title("Scaling: Kernel Time vs. Input Size (V0, V1, V2)")
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt_n(n) for n in xs])
    ax.margins(y=0.15)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_scaling_loglog(sc, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [SCALING_DS_SIZES[ds] for ds in SCALING_DS_ORDER]
    for v in SCALING_VERSIONS:
        ys = [vavg(sc[v][ds]) for ds in SCALING_DS_ORDER]
        mask = [not np.isnan(y) and y > 0 for y in ys]
        xp = [x for x, m in zip(xs, mask) if m]
        yp = [y for y, m in zip(ys, mask) if m]
        if xp:
            ax.plot(xp, yp,
                    marker=VERSION_MARKS[v], color=VERSION_COLORS[v],
                    label=VERSION_LABELS[v], linewidth=1.6, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size $N$ (log scale)")
    ax.set_ylabel("Kernel time (ms, log scale)")
    ax.set_title("Scaling: Log-Log Kernel Time vs. Input Size")
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt_n(n) for n in xs])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_scaling_bandwidth(sc, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [SCALING_DS_SIZES[ds] for ds in SCALING_DS_ORDER]
    for v in SCALING_VERSIONS:
        bws = [calculate_bandwidth(SCALING_DS_SIZES[ds], vavg(sc[v][ds]))
               for ds in SCALING_DS_ORDER]
        mask = [not np.isnan(b) for b in bws]
        xp = [x for x, m in zip(xs, mask) if m]
        bp = [b for b, m in zip(bws, mask) if m]
        if xp:
            ax.plot(xp, bp,
                    marker=VERSION_MARKS[v], color=VERSION_COLORS[v],
                    label=VERSION_LABELS[v], linewidth=1.6, markersize=6)
    ax.set_xlabel("Input size $N$")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("Scaling: Effective Memory Bandwidth vs. Input Size")
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt_n(n) for n in xs])
    ax.margins(y=0.15)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

if SCALING_DIR.exists() and any(SCALING_DIR.iterdir()):
    print("\nGenerating scaling experiment graphs...")
    sc_data = collect_scaling()
    plot_scaling_time(sc_data,      "fig_scaling_time.pdf")
    plot_scaling_loglog(sc_data,    "fig_scaling_loglog.pdf")
    plot_scaling_bandwidth(sc_data, "fig_scaling_bandwidth.pdf")
else:
    print(f"\nScaling data not found in {SCALING_DIR}/ — run sbatch run_scaling.slurm on HPC first.")

print(f"\nAll figures saved to {FIG_DIR}/\n")

# ── LaTeX substitution map ────────────────────────────────────────────────────
NUM_WORDS = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

def build_subs(prefix, times, size_n=500000):
    subs = {}
    for i, word in enumerate(NUM_WORDS):
        subs[f"\\{prefix}{word}"] = fmt(times[i]) if i < len(times) else "N/A"
    a = vavg(times);   subs[f"\\{prefix}Avg"]   = f"{a:.4f}"   if not np.isnan(a) else "N/A"
    s = vstdev(times); subs[f"\\{prefix}Stdev"] = f"{s:.4f}"   if not np.isnan(s) else "N/A"
    bw = calculate_bandwidth(size_n, a)
    subs[f"\\{prefix}BW"] = f"{bw:.2f}" if not np.isnan(bw) else "N/A"
    return subs

substitutions = {}
for prefix, times in [
    ("EOneVZero",  e1[0]), ("EOneVOne",   e1[1]), ("EOneVTwo",   e1[2]), ("EOneVThree", e1[3]),
    ("EOneVFour",  e1[4]), ("EOneVFive",  e1[5]), ("EOneVSix",   e1[6]), ("EOneVSeven", e1[7]),
    ("ETwoVZero",  e2[0]), ("ETwoVOne",   e2[1]), ("ETwoVTwo",   e2[2]), ("ETwoVThree", e2[3]),
    ("ETwoVFour",  e2[4]), ("ETwoVFive",  e2[5]), ("ETwoVSix",   e2[6]), ("ETwoVSeven", e2[7]),
]:
    substitutions.update(build_subs(prefix, times, size_n=500000))

# ── inject into LaTeX ─────────────────────────────────────────────────────────
if "--inject" in sys.argv:
    idx      = sys.argv.index("--inject")
    tex_path = pathlib.Path(sys.argv[idx + 1])
    if not tex_path.is_absolute():
        tex_path = SCRIPT_DIR / tex_path   # resolve relative to report/ dir
    if not tex_path.exists():
        print(f"ERROR: {tex_path} not found.")
        sys.exit(1)

    content  = tex_path.read_text(encoding="utf-8")
    replaced = 0

    def replacer(m):
        global replaced
        cmd = m.group(1)
        if cmd in substitutions:
            replaced += 1
            return f"\\newcommand{{{cmd}}}{{{substitutions[cmd]}}}"
        return m.group(0)

    pattern     = re.compile(r'\\newcommand\{(\\[A-Za-z]+)\}\{[^}]*\}')
    new_content = pattern.sub(replacer, content)
    backup      = tex_path.with_suffix(".tex.bak")
    backup.unlink(missing_ok=True)
    tex_path.rename(backup)
    tex_path.write_text(new_content, encoding="utf-8")
    print(f"Injected {replaced} values into {tex_path}")
    print(f"Backup saved to {backup}")
else:
    print("Tip: run with --inject hw4_report.tex to write values into LaTeX.")
    print("     python3 parse_times.py --inject hw4_report.tex")

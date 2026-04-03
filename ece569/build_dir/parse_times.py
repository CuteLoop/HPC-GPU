#!/usr/bin/env python3
# parse_times.py — ECE569 HW4
# Parses experiment output files, prints timing tables,
# generates plots as PDF, and injects values into hw4_report.tex
#
# Usage (run from build_dir on HPC or locally after git pull):
#   python3 parse_times.py                          # table + graphs only
#   python3 parse_times.py --inject hw4_report.tex  # also injects into LaTeX
#
# Outputs written to Histogram_output/figures/:
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
OUTPUT_ROOT = pathlib.Path("Histogram_output")
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

VERSION_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#3BB273"]
VERSION_LABELS = [
    "V0 — Global Scatter",
    "V1 — Block Privatization",
    "V2 — Warp Aggregation (uint4+RLE)",
    "V3 — Bin-Centric Gather",
]
VERSION_MARKS  = ["o", "s", "^", "D"]
RUNS           = list(range(1, 11))
NUM_VERSIONS   = 4   # V0, V1, V2, V3

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
        text = filepath.read_text()
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
e1 = [collect(EXP1_DIR, v) for v in range(NUM_VERSIONS)]
e2 = [collect(EXP2_DIR, v) for v in range(NUM_VERSIONS)]

# ── print tables ──────────────────────────────────────────────────────────────
def print_table(label, data):
    print(f"{'='*85}")
    print(f"  {label}")
    print(f"{'='*85}")
    print(f"  {'Run':<5} {'V0 (ms)':<14} {'V1 (ms)':<14} {'V2 (ms)':<14} {'V3 (ms)':<14}")
    print(f"  {'-'*60}")
    for i in range(10):
        row = [fmt(data[v][i]) if i < len(data[v]) else "N/A" for v in range(NUM_VERSIONS)]
        print(f"  {i+1:<5} {row[0]:<14} {row[1]:<14} {row[2]:<14} {row[3]:<14}")
    print(f"  {'-'*60}")
    avgs = [f"{vavg(data[v]):.4f}" for v in range(NUM_VERSIONS)]
    print(f"  {'Avg':<5} {avgs[0]:<14} {avgs[1]:<14} {avgs[2]:<14} {avgs[3]:<14}")
    print()

print_table("Experiment 1 — Random data (dataset 6, 500k elements)", e1)
print_table("Experiment 2 — Uniform data (500k same-value elements)", e2)

# ════════════════════════════════════════════════════════════
#  FIG 1 & 2 — Run-by-run line charts
# ════════════════════════════════════════════════════════════
def plot_runs(data, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in range(NUM_VERSIONS):
        ax.plot(RUNS, nan_list(data[v]),
                marker=VERSION_MARKS[v],
                color=VERSION_COLORS[v],
                label=VERSION_LABELS[v],
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

plot_runs(e1, "Experiment 1 — Kernel time per run (random data)",  "fig_exp1_runs.pdf")
plot_runs(e2, "Experiment 2 — Kernel time per run (uniform data)", "fig_exp2_runs.pdf")

# ════════════════════════════════════════════════════════════
#  FIG 3 & 4 — Average bar charts with std dev error bars
# ════════════════════════════════════════════════════════════
XLABELS = ["V0\nGlobal\nScatter", "V1\nBlock\nPrivatization", "V2\nWarp\nAggregation", "V3\nBin-Centric\nGather"]

def plot_avg_bar(data, title, filename):
    avgs = [vavg(data[v])   for v in range(NUM_VERSIONS)]
    errs = [vstdev(data[v]) for v in range(NUM_VERSIONS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(NUM_VERSIONS)
    bars = ax.bar(x, avgs, yerr=errs,
                  color=VERSION_COLORS, width=0.5,
                  capsize=5, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    ax.set_ylabel("Average kernel time (ms)")
    ax.set_title(title)
    err_max = max(e for e in errs if not np.isnan(e)) if errs else 0
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
fig, ax = plt.subplots(figsize=(8, 4.5))
x      = np.arange(NUM_VERSIONS)
width  = 0.32
avgs1  = [vavg(e1[v])   for v in range(NUM_VERSIONS)]
avgs2  = [vavg(e2[v])   for v in range(NUM_VERSIONS)]
errs1  = [vstdev(e1[v]) for v in range(NUM_VERSIONS)]
errs2  = [vstdev(e2[v]) for v in range(NUM_VERSIONS)]

ax.bar(x - width/2, avgs1, width, yerr=errs1,
       color=VERSION_COLORS, alpha=0.90,
       capsize=4, error_kw={"linewidth": 1.1})
ax.bar(x + width/2, avgs2, width, yerr=errs2,
       color=VERSION_COLORS, alpha=0.45,
       capsize=4, error_kw={"linewidth": 1.1}, hatch="//")

ax.legend(handles=[
    Patch(facecolor="#888", alpha=0.90, label="Experiment 1 — random"),
    Patch(facecolor="#888", alpha=0.45, hatch="//", label="Experiment 2 — uniform"),
], loc="upper right")
ax.set_xticks(x)
ax.set_xticklabels(XLABELS)
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
def plot_speedup(data, title, filename):
    baseline = vavg(data[0])
    if np.isnan(baseline) or baseline == 0:
        print(f"  SKIP speedup chart (no V0 baseline): {filename}")
        return
    speedups = [baseline / vavg(data[v]) if not np.isnan(vavg(data[v])) and vavg(data[v]) > 0
                else float("nan") for v in range(NUM_VERSIONS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(NUM_VERSIONS)
    bars = ax.bar(x, speedups, color=VERSION_COLORS, width=0.5)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5, label="V0 baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    ax.set_ylabel("Speedup relative to V0")
    ax.set_title(title)
    ax.legend(loc="upper left")
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
plot_speedup(e2, "Experiment 2 — Speedup relative to V0 (uniform data)", "fig_exp2_speedup.pdf")

print(f"\nAll figures saved to {FIG_DIR}/\n")

# ── LaTeX substitution map ────────────────────────────────────────────────────
NUM_WORDS = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

def build_subs(prefix, times):
    subs = {}
    for i, word in enumerate(NUM_WORDS):
        subs[f"\\{prefix}{word}"] = fmt(times[i]) if i < len(times) else "N/A"
    a = vavg(times);   subs[f"\\{prefix}Avg"]   = f"{a:.4f}"   if not np.isnan(a) else "N/A"
    s = vstdev(times); subs[f"\\{prefix}Stdev"] = f"{s:.4f}"   if not np.isnan(s) else "N/A"
    return subs

substitutions = {}
for prefix, times in [
    ("EOneVZero", e1[0]), ("EOneVOne", e1[1]), ("EOneVTwo", e1[2]), ("EOneVThree", e1[3]),
    ("ETwoVZero", e2[0]), ("ETwoVOne", e2[1]), ("ETwoVTwo", e2[2]), ("ETwoVThree", e2[3]),
]:
    substitutions.update(build_subs(prefix, times))

# ── inject into LaTeX ─────────────────────────────────────────────────────────
if "--inject" in sys.argv:
    idx      = sys.argv.index("--inject")
    tex_path = pathlib.Path(sys.argv[idx + 1])
    if not tex_path.exists():
        print(f"ERROR: {tex_path} not found.")
        sys.exit(1)

    content  = tex_path.read_text()
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
    tex_path.rename(backup)
    tex_path.write_text(new_content)
    print(f"Injected {replaced} values into {tex_path}")
    print(f"Backup saved to {backup}")
else:
    print("Tip: run with --inject hw4_report.tex to write values into LaTeX.")
    print("     python3 parse_times.py --inject hw4_report.tex")

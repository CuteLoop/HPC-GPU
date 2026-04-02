#!/usr/bin/env python3
"""Create combined scaling plots for Basic and Tiled kernels.

Combines two datasets:
1) First-run outputs:
   - ece569/build_dir/BasicMatrixMultiplication_output/output0..9.txt
   - ece569/build_dir/TiledMatrixMultiplication_output/output0..9.txt
2) Performance-analysis outputs:
   - ece569/build_dir/BasicPerformance_output/bs*_case*_run*.txt
   - ece569/build_dir/TiledPerformance_output/bs*_case*_run*.txt

Produces:
- graph1_basic_times_vs_size_combined.(png|pdf)
- graph2_tiled_times_vs_size_combined.(png|pdf)

X-axis uses workload proxy M*N*K (from logged matrix dimensions).
Y-axis is compute time in ms.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

DIMS_RE = re.compile(r"(\d+)\s*x\s*(\d+)")
PERF_NAME_RE = re.compile(r"^bs(?P<block>\d+)_case(?P<case>\d+)_run(?P<run>\d+)\.txt$")


@dataclass
class Point:
    x_workload: int
    y_ms: float
    label: str


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "figure.facecolor": "white",
            "axes.facecolor": "#FAF9F7",
            "axes.edgecolor": "#3D3D3D",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": True,
        }
    )


def parse_json_lines(file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    a_rows = a_cols = b_cols = None
    compute_ns = None

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") == "logger":
                msg = str(obj.get("data", {}).get("message", ""))
                m = DIMS_RE.search(msg)
                if m:
                    r, c = int(m.group(1)), int(m.group(2))
                    if "dimensions of A" in msg:
                        a_rows = r
                        a_cols = c
                    elif "dimensions of B" in msg:
                        b_cols = c

            elif obj.get("type") == "timer":
                data = obj.get("data", {})
                if data.get("kind") == "Compute":
                    compute_ns = int(data.get("elapsed_time", 0))

    return a_rows, a_cols, b_cols, compute_ns


def parse_first_run(folder: Path, prefix: str) -> List[Point]:
    points: List[Point] = []
    files = sorted(folder.glob("output*.txt"), key=lambda p: int(p.stem.replace("output", "")))

    for idx, fp in enumerate(files):
        a_rows, a_cols, b_cols, compute_ns = parse_json_lines(fp)
        if None in (a_rows, a_cols, b_cols, compute_ns):
            continue
        workload = int(a_rows * a_cols * b_cols)
        ms = float(compute_ns) / 1_000_000.0
        points.append(Point(workload, ms, f"{prefix}{idx}"))

    points.sort(key=lambda p: p.x_workload)
    return points


def parse_performance(folder: Path) -> Dict[int, List[Point]]:
    grouped: Dict[Tuple[int, int], List[Tuple[int, float, str]]] = {}

    for fp in sorted(folder.glob("*.txt")):
        m = PERF_NAME_RE.match(fp.name)
        if not m:
            continue
        block = int(m.group("block"))
        case_id = int(m.group("case"))

        a_rows, a_cols, b_cols, compute_ns = parse_json_lines(fp)
        if None in (a_rows, a_cols, b_cols, compute_ns):
            continue
        workload = int(a_rows * a_cols * b_cols)
        ms = float(compute_ns) / 1_000_000.0
        grouped.setdefault((block, case_id), []).append((workload, ms, f"case{case_id}"))

    out: Dict[int, List[Point]] = {}
    by_block: Dict[int, List[Point]] = {}

    for (block, _case_id), vals in grouped.items():
        workload = vals[0][0]
        label = vals[0][2]
        avg_ms = mean(v[1] for v in vals)
        by_block.setdefault(block, []).append(Point(workload, avg_ms, label))

    for block, pts in by_block.items():
        pts.sort(key=lambda p: p.x_workload)
        out[block] = pts

    return out


def save_plot(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def build_plot(
    title: str,
    first_run_points: List[Point],
    perf_by_block: Dict[int, List[Point]],
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))

    x_fr = [p.x_workload for p in first_run_points]
    y_fr = [p.y_ms for p in first_run_points]
    ax.plot(x_fr, y_fr, marker="o", linewidth=2.0, color="#1F3A5F", label="First-run cases (output0-9)")

    palette = ["#4C956C", "#D17B0F", "#8F2D56", "#5E81AC", "#2B2D42"]
    for i, block in enumerate(sorted(perf_by_block.keys())):
        pts = perf_by_block[block]
        ax.plot(
            [p.x_workload for p in pts],
            [p.y_ms for p in pts],
            marker="s",
            linewidth=1.9,
            color=palette[i % len(palette)],
            label=f"Performance cases (bs{block})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem Size Proxy (M*N*K), log scale")
    ax.set_ylabel("Compute Time (ms), log scale")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    save_plot(fig, out_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined Basic/Tiled scaling plots.")
    parser.add_argument("--basic-first-dir", type=Path, default=Path("ece569/build_dir/BasicMatrixMultiplication_output"))
    parser.add_argument("--tiled-first-dir", type=Path, default=Path("ece569/build_dir/TiledMatrixMultiplication_output"))
    parser.add_argument("--basic-perf-dir", type=Path, default=Path("ece569/build_dir/BasicPerformance_output"))
    parser.add_argument("--tiled-perf-dir", type=Path, default=Path("ece569/build_dir/TiledPerformance_output"))
    parser.add_argument("--out-dir", type=Path, default=Path("ece569/build_dir/perf_analysis_graphs"))
    args = parser.parse_args()

    configure_style()

    basic_first = parse_first_run(args.basic_first_dir, "b")
    tiled_first = parse_first_run(args.tiled_first_dir, "t")
    basic_perf = parse_performance(args.basic_perf_dir)
    tiled_perf = parse_performance(args.tiled_perf_dir)

    if not basic_first or not tiled_first:
        raise RuntimeError("First-run output parsing failed. Check output0..9 files.")
    if not basic_perf or not tiled_perf:
        raise RuntimeError("Performance output parsing failed. Check bs*_case*_run*.txt files.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    build_plot(
        title="Graph 1: Basic Multiplication Time vs Problem Size (First-run + Performance Cases)",
        first_run_points=basic_first,
        perf_by_block=basic_perf,
        out_base=args.out_dir / "graph1_basic_times_vs_size_combined",
    )

    build_plot(
        title="Graph 2: Tiled Multiplication Time vs Problem Size (First-run + Performance Cases)",
        first_run_points=tiled_first,
        perf_by_block=tiled_perf,
        out_base=args.out_dir / "graph2_tiled_times_vs_size_combined",
    )

    print(f"Saved combined plots to: {args.out_dir}")


if __name__ == "__main__":
    main()

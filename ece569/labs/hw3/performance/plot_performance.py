#!/usr/bin/env python3
"""Generate required performance-analysis graphs for HW3 matrix multiplication.

This script parses JSON-lines output files from:
- BasicPerformance_output/
- TiledPerformance_output/

Expected filename pattern:
    bs<block>_case<case>_run<run>.txt

It extracts only the "Compute" timer and averages execution time over runs.
The script produces:
1) Basic version scaling line chart
2) Tiled version scaling line chart
3) Clustered bar chart (Basic vs Tiled) for the largest input size

It also writes a CSV summary with mean/std/count per (kernel, case, block size).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

FILE_RE = re.compile(r"^bs(?P<block>\d+)_case(?P<case>\d+)_run(?P<run>\d+)\.txt$")
DIMS_RE = re.compile(r"(\d+)\s*x\s*(\d+)")


LINE_COLORS = ["#1F3A5F", "#4C956C", "#D17B0F", "#8F2D56", "#2B2D42"]
BAR_COLORS = {"Basic": "#5E81AC", "Tiled": "#A3BE8C"}


def configure_plot_style() -> None:
    """Apply a print-friendly style that blends well with LaTeX reports."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "CMU Serif",
                "Computer Modern Roman",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#FAF9F7",
            "axes.edgecolor": "#3D3D3D",
            "axes.linewidth": 0.9,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "grid.color": "#CCC7BD",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#BFBFBF",
            "legend.framealpha": 1.0,
            "xtick.color": "#2F2F2F",
            "ytick.color": "#2F2F2F",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
        }
    )


def save_figure(base_path: Path, dpi: int = 220) -> None:
    """Save each figure as both PNG and PDF for report usage."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


@dataclass(frozen=True)
class CaseInfo:
    case_id: int
    a_rows: Optional[int] = None
    a_cols: Optional[int] = None
    b_rows: Optional[int] = None
    b_cols: Optional[int] = None

    @property
    def c_rows(self) -> Optional[int]:
        return self.a_rows

    @property
    def c_cols(self) -> Optional[int]:
        return self.b_cols

    @property
    def complexity_score(self) -> float:
        """Use M*N*K as a proxy for compute workload."""
        if None in (self.a_rows, self.a_cols, self.b_cols):
            return float(self.case_id)
        return float(self.a_rows * self.a_cols * self.b_cols)

    @property
    def label(self) -> str:
        if self.c_rows is None or self.c_cols is None:
            return f"Case {self.case_id}"
        return f"Case {self.case_id} ({self.c_rows}x{self.c_cols})"


@dataclass
class Record:
    kernel: str
    case_id: int
    block_size: int
    run_id: int
    compute_ms: float


def _extract_dims(message: str) -> Optional[Tuple[int, int]]:
    match = DIMS_RE.search(message)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_output_file(file_path: Path, kernel_name: str) -> Tuple[Optional[Record], Optional[CaseInfo]]:
    name_match = FILE_RE.match(file_path.name)
    if not name_match:
        return None, None

    block_size = int(name_match.group("block"))
    case_id = int(name_match.group("case"))
    run_id = int(name_match.group("run"))

    compute_elapsed_ns: Optional[int] = None
    dims: Dict[str, Tuple[int, int]] = {}

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") == "timer":
                data = obj.get("data", {})
                if data.get("kind") == "Compute":
                    compute_elapsed_ns = int(data["elapsed_time"])

            if obj.get("type") == "logger":
                data = obj.get("data", {})
                msg = str(data.get("message", ""))
                if "dimensions of A" in msg:
                    parsed = _extract_dims(msg)
                    if parsed:
                        dims["A"] = parsed
                elif "dimensions of B" in msg:
                    parsed = _extract_dims(msg)
                    if parsed:
                        dims["B"] = parsed

    if compute_elapsed_ns is None:
        return None, None

    rec = Record(
        kernel=kernel_name,
        case_id=case_id,
        block_size=block_size,
        run_id=run_id,
        compute_ms=compute_elapsed_ns / 1_000_000.0,
    )

    case_info = CaseInfo(
        case_id=case_id,
        a_rows=dims.get("A", (None, None))[0],
        a_cols=dims.get("A", (None, None))[1],
        b_rows=dims.get("B", (None, None))[0],
        b_cols=dims.get("B", (None, None))[1],
    )

    return rec, case_info


def collect_records(folder: Path, kernel_name: str) -> Tuple[List[Record], Dict[int, CaseInfo]]:
    records: List[Record] = []
    case_info: Dict[int, CaseInfo] = {}

    for file_path in sorted(folder.glob("*.txt")):
        rec, info = parse_output_file(file_path, kernel_name)
        if rec is None:
            continue

        records.append(rec)

        if info is not None:
            existing = case_info.get(info.case_id)
            if existing is None:
                case_info[info.case_id] = info
            else:
                # Prefer the variant that has dimension metadata populated.
                if existing.c_rows is None and info.c_rows is not None:
                    case_info[info.case_id] = info

    return records, case_info


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def sample_std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def aggregate(records: List[Record]) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    grouped: Dict[Tuple[str, int, int], List[float]] = {}

    for rec in records:
        key = (rec.kernel, rec.case_id, rec.block_size)
        grouped.setdefault(key, []).append(rec.compute_ms)

    out: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for key, vals in grouped.items():
        out[key] = {
            "mean_ms": mean(vals),
            "std_ms": sample_std(vals),
            "count": float(len(vals)),
        }

    return out


def write_summary_csv(summary: Dict[Tuple[str, int, int], Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, int, int, float, float, int]] = []
    for (kernel, case_id, block_size), stats in summary.items():
        rows.append(
            (
                kernel,
                case_id,
                block_size,
                stats["mean_ms"],
                stats["std_ms"],
                int(stats["count"]),
            )
        )

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "case_id", "block_size", "mean_ms", "std_ms", "num_runs"])
        writer.writerows(rows)


def _ordered_case_ids(case_info: Dict[int, CaseInfo], summary: Dict[Tuple[str, int, int], Dict[str, float]]) -> List[int]:
    known = set(case_info.keys())
    known.update(case for (_, case, _) in summary.keys())
    return sorted(known)


def _ordered_block_sizes(summary: Dict[Tuple[str, int, int], Dict[str, float]]) -> List[int]:
    return sorted({block for (_, _, block) in summary.keys()})


def _case_display_labels(case_ids: List[int], case_info: Dict[int, CaseInfo]) -> List[str]:
    labels: List[str] = []
    for idx, cid in enumerate(case_ids, start=1):
        info = case_info.get(cid, CaseInfo(case_id=cid))
        if info.c_rows is None or info.c_cols is None:
            labels.append(f"Case {idx}")
        else:
            labels.append(f"Case {idx} ({info.c_rows}x{info.c_cols})")
    return labels


def plot_scaling(
    summary: Dict[Tuple[str, int, int], Dict[str, float]],
    case_info: Dict[int, CaseInfo],
    kernel: str,
    out_path: Path,
) -> None:
    case_ids = _ordered_case_ids(case_info, summary)
    block_sizes = _ordered_block_sizes(summary)

    labels = _case_display_labels(case_ids, case_info)
    x = list(range(len(case_ids)))

    plt.figure(figsize=(10, 5.6))
    for i, bs in enumerate(block_sizes):
        y = [summary.get((kernel, cid, bs), {}).get("mean_ms", float("nan")) for cid in case_ids]
        plt.plot(x, y, marker="o", color=LINE_COLORS[i % len(LINE_COLORS)], label=f"{bs}x{bs}")

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Execution Time (ms)")
    plt.xlabel("Problem Size")
    plt.title(f"{kernel} Version Scaling")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Block Size")
    plt.tight_layout()

    save_figure(out_path)
    plt.close()


def pick_largest_case(case_info: Dict[int, CaseInfo], summary: Dict[Tuple[str, int, int], Dict[str, float]]) -> int:
    case_ids = _ordered_case_ids(case_info, summary)
    best_case = case_ids[0]
    best_score = -1.0

    for cid in case_ids:
        score = case_info.get(cid, CaseInfo(case_id=cid)).complexity_score
        if score > best_score:
            best_score = score
            best_case = cid

    return best_case


def plot_largest_case_bar(
    summary: Dict[Tuple[str, int, int], Dict[str, float]],
    case_info: Dict[int, CaseInfo],
    out_path: Path,
) -> None:
    block_sizes = _ordered_block_sizes(summary)
    largest_case = pick_largest_case(case_info, summary)
    case_ids = _ordered_case_ids(case_info, summary)
    case_idx = case_ids.index(largest_case) + 1

    basic_vals = [summary.get(("Basic", largest_case, bs), {}).get("mean_ms", float("nan")) for bs in block_sizes]
    tiled_vals = [summary.get(("Tiled", largest_case, bs), {}).get("mean_ms", float("nan")) for bs in block_sizes]

    x = list(range(len(block_sizes)))
    width = 0.38

    plt.figure(figsize=(9.6, 5.6))
    plt.bar(
        [i - width / 2 for i in x],
        basic_vals,
        width=width,
        label="Basic",
        color=BAR_COLORS["Basic"],
        edgecolor="#2F2F2F",
        linewidth=0.6,
    )
    plt.bar(
        [i + width / 2 for i in x],
        tiled_vals,
        width=width,
        label="Tiled",
        color=BAR_COLORS["Tiled"],
        edgecolor="#2F2F2F",
        linewidth=0.6,
    )

    plt.xticks(x, [f"{bs}x{bs}" for bs in block_sizes])
    plt.ylabel("Execution Time (ms)")
    plt.xlabel("Block Size Configuration")

    case_label = case_info.get(largest_case, CaseInfo(case_id=largest_case))
    if case_label.c_rows is not None and case_label.c_cols is not None:
        title_case = f"Case {case_idx} ({case_label.c_rows}x{case_label.c_cols})"
    else:
        title_case = f"Case {case_idx}"
    plt.title(f"Largest Input Comparison: {title_case}")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_figure(out_path)
    plt.close()


def plot_speedup_heatmap(
    summary: Dict[Tuple[str, int, int], Dict[str, float]],
    case_info: Dict[int, CaseInfo],
    out_path: Path,
) -> None:
    """Plot Basic/Tiled speedup (values > 1 mean tiled is faster)."""
    case_ids = _ordered_case_ids(case_info, summary)
    block_sizes = _ordered_block_sizes(summary)

    matrix: List[List[float]] = []
    for cid in case_ids:
        row: List[float] = []
        for bs in block_sizes:
            basic = summary.get(("Basic", cid, bs), {}).get("mean_ms", float("nan"))
            tiled = summary.get(("Tiled", cid, bs), {}).get("mean_ms", float("nan"))
            row.append(basic / tiled if tiled and not math.isnan(tiled) else float("nan"))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=4.0)

    ax.set_xticks(list(range(len(block_sizes))))
    ax.set_xticklabels([f"{bs}x{bs}" for bs in block_sizes])
    ax.set_yticks(list(range(len(case_ids))))
    ax.set_yticklabels(_case_display_labels(case_ids, case_info))
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Problem Size")
    ax.set_title("Speedup Heatmap: Basic/Tiled (>1 means Tiled faster)")

    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            if math.isnan(val):
                continue
            ax.text(c, r, f"{val:.2f}x", ha="center", va="center", fontsize=9, color="#1F1F1F")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Speedup (Basic time / Tiled time)")
    fig.tight_layout()

    save_figure(out_path)
    plt.close()


def merge_case_info(a: Dict[int, CaseInfo], b: Dict[int, CaseInfo]) -> Dict[int, CaseInfo]:
    merged = dict(a)
    for cid, info in b.items():
        prev = merged.get(cid)
        if prev is None:
            merged[cid] = info
            continue
        if prev.c_rows is None and info.c_rows is not None:
            merged[cid] = info
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matrix multiplication performance graphs.")
    parser.add_argument(
        "--basic-dir",
        type=Path,
        default=Path("ece569/build_dir/BasicPerformance_output"),
        help="Path to BasicPerformance_output directory",
    )
    parser.add_argument(
        "--tiled-dir",
        type=Path,
        default=Path("ece569/build_dir/TiledPerformance_output"),
        help="Path to TiledPerformance_output directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ece569/build_dir/perf_analysis_graphs"),
        help="Directory where graphs and CSV summary are written",
    )

    args = parser.parse_args()
    configure_plot_style()

    if not args.basic_dir.exists():
        raise FileNotFoundError(f"Basic output directory not found: {args.basic_dir}")
    if not args.tiled_dir.exists():
        raise FileNotFoundError(f"Tiled output directory not found: {args.tiled_dir}")

    basic_records, basic_info = collect_records(args.basic_dir, "Basic")
    tiled_records, tiled_info = collect_records(args.tiled_dir, "Tiled")

    all_records = basic_records + tiled_records
    if not all_records:
        raise RuntimeError("No valid records found. Check input paths and file format.")

    summary = aggregate(all_records)
    case_info = merge_case_info(basic_info, tiled_info)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(summary, args.out_dir / "summary.csv")

    plot_scaling(
        summary=summary,
        case_info=case_info,
        kernel="Basic",
        out_path=args.out_dir / "plot1_basic_scaling.png",
    )

    plot_scaling(
        summary=summary,
        case_info=case_info,
        kernel="Tiled",
        out_path=args.out_dir / "plot2_tiled_scaling.png",
    )

    plot_largest_case_bar(
        summary=summary,
        case_info=case_info,
        out_path=args.out_dir / "plot3_largest_case_basic_vs_tiled.png",
    )

    plot_speedup_heatmap(
        summary=summary,
        case_info=case_info,
        out_path=args.out_dir / "plot4_speedup_heatmap.png",
    )

    print(f"Saved plots and summary to: {args.out_dir}")


if __name__ == "__main__":
    main()

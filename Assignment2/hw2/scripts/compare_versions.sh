#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Compare 3 versions and print a small summary table.
#
# Assumptions:
#  - Your GPU executables are profiled with nvprof and have ONE kernel line
#    under "GPU activities" (e.g., add(int, float*, float*))
#  - Your CPU executable prints a time in nanoseconds somewhere like:
#        Time: 91811206 ns
#    (If it prints differently, adjust CPU_TIME_REGEX below.)
#  - All versions use the same N (= 1<<20 by default).
#
# Usage:
#   module load cuda11/11.0
#   chmod +x compare_versions.sh
#   ./compare_versions.sh
# ============================================================

# ---- EDIT THESE ----
EXES=("add" "add_block" "add_grid")   # executable names (in current dir)
LABELS=("Single Thread" "Single Block (256 threads)" "Multiple Blocks")
N=$((1<<20))                              # must match your code
# --------------------

# bytes moved for y[i] = x[i] + y[i] with float:
# read x (4) + read y (4) + write y (4) = 12 bytes per element
BYTES=$((3 * N * 4))

# For CPU executable output parsing (adjust if your CPU prints differently)
CPU_TIME_REGEX='([0-9]+)[[:space:]]*ns'

# Convert nvprof time token -> integer ns (supports ns/us/ms/s)
to_ns() {
  local token="$1"
  local value unit
  value="$(echo "$token" | sed -E 's/^([0-9]*\.?[0-9]+).*/\1/')"
  unit="$(echo "$token"  | sed -E 's/^[0-9]*\.?[0-9]+(.*)/\1/')"

  case "$unit" in
    ns) awk "BEGIN{printf \"%.0f\", $value}" ;;
    us) awk "BEGIN{printf \"%.0f\", $value * 1000}" ;;
    ms) awk "BEGIN{printf \"%.0f\", $value * 1000000}" ;;
    s)  awk "BEGIN{printf \"%.0f\", $value * 1000000000}" ;;
    *)  echo "" ;;
  esac
}

# Extract kernel time token from nvprof log (Time column for first kernel row)
extract_nvprof_kernel_token() {
  local log="$1"
  # After "GPU activities:" header, first non-empty line is the kernel row.
  awk '/GPU activities:/{flag=1;next} flag && NF{print $3; exit}' "$log" 2>/dev/null || true
}

# Run executable and return time in ns:
#  - If nvprof sees GPU activities, use kernel time
#  - Otherwise parse CPU time from program output
get_time_ns() {
  local exe="$1"
  local log="prof_${exe}.txt"
  local out="run_${exe}.txt"

  # Ensure CUDA tools loaded (no-op if already loaded)
  module load cuda11/11.0 >/dev/null 2>&1 || true

  # Run under nvprof; it will still run CPU-only apps, but may show no GPU activities.
  nvprof --log-file "$log" ./"$exe" >"$out" 2>&1 || true

  local tok ns
  tok="$(extract_nvprof_kernel_token "$log")"
  if [[ -n "${tok:-}" ]]; then
    ns="$(to_ns "$tok")"
    [[ -n "${ns:-}" ]] && { echo "$ns"; return 0; }
  fi

  # CPU fallback: find first "<int> ns" in stdout/stderr
  ns="$(grep -Eo "$CPU_TIME_REGEX" "$out" | head -n1 | awk '{print $1}' || true)"
  if [[ -z "${ns:-}" ]]; then
    echo "ERROR: Could not extract time for $exe." >&2
    echo " - GPU: expected nvprof GPU activities line." >&2
    echo " - CPU: expected program to print something like 'Time: <int> ns'." >&2
    exit 1
  fi
  echo "$ns"
}

# Pretty-print ns to ns/us/ms
pretty_time() {
  local t="$1"
  if (( t < 1000000 )); then
    echo "${t} ns"
  elif (( t < 1000000000 )); then
    awk "BEGIN{printf \"%.3f us\", $t/1000}"
  else
    awk "BEGIN{printf \"%.3f ms\", $t/1000000}"
  fi
}

# Main: collect times
times_ns=()
for exe in "${EXES[@]}"; do
  times_ns+=("$(get_time_ns "$exe")")
done

T0="${times_ns[0]}"

# Print table
printf "%-26s  %14s  %16s  %12s\n" "Version" "Time" "Speedup vs CPU" "Bandwidth"
printf "%-26s  %14s  %16s  %12s\n" "--------------------------" "--------------" "----------------" "----------"

for i in "${!EXES[@]}"; do
  t="${times_ns[$i]}"

  # speedup
  speed="$(awk "BEGIN{printf \"%.1fx\", $T0/$t}")"

  # bandwidth: bytes/ns == GB/s numerically (since 1 GB/s = 1 byte/ns)
  bw="$(awk "BEGIN{printf \"%.1f GB/s\", $BYTES/$t}")"

  printf "%-26s  %14s  %16s  %12s\n" "${LABELS[$i]}" "$(pretty_time "$t")" "$speed" "$bw"
done

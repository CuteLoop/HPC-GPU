#!/usr/bin/env bash
set -euo pipefail

# ===== EDIT THESE if needed =====
CPU_EXE="add_cpu"
GPU_EXES=("add, add_block" "add_grid")
LABELS=("Single Thread" "Single Block (256 threads)" "Multiple Blocks")
N=$((1<<20))                    # must match code
# ================================

BYTES=$((3 * N * 4))            # 12 bytes/element for y = x + y (float)

module load cuda11/11.0 >/dev/null 2>&1 || true

# Convert nvprof time token -> ns (ns/us/ms/s)
to_ns() {
  local tok="$1"
  local val unit
  val="$(echo "$tok" | sed -E 's/^([0-9]*\.?[0-9]+).*/\1/')"
  unit="$(echo "$tok" | sed -E 's/^[0-9]*\.?[0-9]+(.*)/\1/')"
  case "$unit" in
    ns) awk "BEGIN{printf \"%.0f\", $val}" ;;
    us) awk "BEGIN{printf \"%.0f\", $val * 1000}" ;;
    ms) awk "BEGIN{printf \"%.0f\", $val * 1000000}" ;;
    s)  awk "BEGIN{printf \"%.0f\", $val * 1000000000}" ;;
    *)  echo "" ;;
  esac
}

# Pretty-print ns
pretty_time() {
  local ns="$1"
  if (( ns < 1000000 )); then
    echo "${ns} ns"
  elif (( ns < 1000000000 )); then
    awk "BEGIN{printf \"%.3f us\", $ns/1000}"
  else
    awk "BEGIN{printf \"%.3f ms\", $ns/1000000}"
  fi
}

# CPU wall-time in ns using /usr/bin/time (no code changes needed)
cpu_time_ns() {
  local exe="$1"
  local secs
  # /usr/bin/time prints to stderr; capture it. %e = elapsed real time in seconds (decimal)
  secs="$(/usr/bin/time -f "%e" ./"$exe" >/dev/null 2>&1 2>&1)"
  # convert seconds -> ns
  awk "BEGIN{printf \"%.0f\", $secs * 1000000000}"
}

# nvprof kernel time in ns (GPU activities)
gpu_kernel_time_ns() {
  local exe="$1"
  local log="prof_${exe}.txt"
  nvprof --log-file "$log" ./"$exe" >/dev/null 2>&1 || true
  local tok
  tok="$(awk '/GPU activities:/{f=1;next} f && NF{print $3; exit}' "$log" 2>/dev/null || true)"
  if [[ -z "${tok:-}" ]]; then
    echo "ERROR: Could not find GPU kernel time for $exe in nvprof output." >&2
    echo "Hint: run 'nvprof ./$exe' manually and confirm there is a GPU activities kernel row." >&2
    exit 1
  fi
  to_ns "$tok"
}

# ---- collect times ----
t_cpu="$(cpu_time_ns "$CPU_EXE")"
t_block="$(gpu_kernel_time_ns "${GPU_EXES[0]}")"
t_grid="$(gpu_kernel_time_ns "${GPU_EXES[1]}")"

times=("$t_cpu" "$t_block" "$t_grid")

# ---- print table ----
printf "%-26s  %14s  %16s  %12s\n" "Version" "Time" "Speedup vs CPU" "Bandwidth"
printf "%-26s  %14s  %16s  %12s\n" "--------------------------" "--------------" "----------------" "----------"

T0="${times[0]}"

for i in 0 1 2; do
  t="${times[$i]}"
  speed="$(awk "BEGIN{printf \"%.1fx\", $T0/$t}")"
  bw="$(awk "BEGIN{printf \"%.1f GB/s\", $BYTES/$t}")"   # bytes/ns == GB/s
  printf "%-26s  %14s  %16s  %12s\n" "${LABELS[$i]}" "$(pretty_time "$t")" "$speed" "$bw"
done

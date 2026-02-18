#!/usr/bin/env bash
set -euo pipefail

# ===== EDIT THESE if needed =====
CPU_EXE="add_cpu"
GPU_EXES=("add_block" "add_grid")
LABELS=("Single Thread" "Single Block (256 threads)" "Multiple Blocks")
N=$((1<<20))
# ================================

# bytes moved for y[i] = x[i] + y[i] with float: 12 bytes/element
BYTES=$((3 * N * 4))

module load cuda11/11.0 >/dev/null 2>&1 || true

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

cpu_time_ns() {
  local exe="$1"
  local out secs
  out="$(mktemp)"

  # GNU time -p prints:
  # real <sec>
  # user <sec>
  # sys  <sec>
  /usr/bin/time -p ./"$exe" >/dev/null 2>"$out" || true

  secs="$(awk '/^real[[:space:]]+/ {print $2; exit}' "$out")"
  rm -f "$out"

  if [[ -z "${secs:-}" ]]; then
    echo "ERROR: Could not parse CPU time. Ensure /usr/bin/time exists and $exe runs." >&2
    exit 1
  fi

  # seconds -> ns
  awk "BEGIN{printf \"%.0f\", $secs * 1000000000}"
}

gpu_kernel_time_ns() {
  local exe="$1"
  local log="prof_${exe}.txt"

  nvprof --log-file "$log" ./"$exe" >/dev/null 2>&1 || true

  # nvprof can place the kernel row either:
  #  (A) on the SAME LINE as "GPU activities:"  (your case), or
  #  (B) on the next line(s)
  #
  # So: scan from "GPU activities:" up to "API calls:" and grab the first token
  # that looks like a time:  number + (ns|us|ms|s)
  local tok
  tok="$(awk '
    /GPU activities:/ {in=1}
    in {
      if ($0 ~ /API calls:/) exit
      for (i=1; i<=NF; i++) {
        gsub(/\r/, "", $i)
        if ($i ~ /^[0-9]*\.?[0-9]+(ns|us|ms|s)$/) { print $i; exit }
      }
    }
  ' "$log" 2>/dev/null || true)"

  if [[ -z "${tok:-}" ]]; then
    echo "ERROR: Could not find a kernel time token for $exe in $log." >&2
    echo "Debug: sed -n '/GPU activities:/,/API calls:/p' $log" >&2
    exit 1
  fi

  local ns
  ns="$(to_ns "$tok")"
  if [[ -z "${ns:-}" ]]; then
    echo "ERROR: Could not parse nvprof time token '$tok' for $exe." >&2
    exit 1
  fi
  echo "$ns"
}

# ---- collect times ----
t_cpu="$(cpu_time_ns "$CPU_EXE")"
t_block="$(gpu_kernel_time_ns "${GPU_EXES[0]}")"
t_grid="$(gpu_kernel_time_ns "${GPU_EXES[1]}")"

times=("$t_cpu" "$t_block" "$t_grid")
T0="${times[0]}"

# ---- print table ----
printf "%-26s  %14s  %16s  %12s\n" "Version" "Time" "Speedup vs CPU" "Bandwidth"
printf "%-26s  %14s  %16s  %12s\n" "--------------------------" "--------------" "----------------" "----------"

for i in 0 1 2; do
  t="${times[$i]}"
  speed="$(awk "BEGIN{printf \"%.1fx\", $T0/$t}")"
  bw="$(awk "BEGIN{printf \"%.1f GB/s\", $BYTES/$t}")"   # bytes/ns == GB/s
  printf "%-26s  %14s  %16s  %12s\n" "${LABELS[$i]}" "$(pretty_time "$t")" "$speed" "$bw"
done

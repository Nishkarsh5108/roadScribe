#!/usr/bin/env bash
# Push queued P6 kernels when a GPU slot frees; download outputs of finished kernels.
K="/c/Users/nishk/anaconda3/python.exe -m kaggle"
cd "D:/Coding/Hackathon/GFG/ARM/kaggle_staging"
OUTD="D:/Coding/Hackathon/GFG/ARM/roadScribe/data/models/p6"
mkdir -p "$OUTD"
declare -A SLUG=( [B]=roadscribe-p6-arm-b [C]=roadscribe-p6-arm-c [A]=roadscribe-p6-arm-a [zs]=roadscribe-p6-zeroshot )
running=(B C)
queue=(A zs)
done_list=()
status() { $K kernels status "nishkarshsinghal/${SLUG[$1]}" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | sed 's/KernelWorkerStatus\.//'; }
while [ ${#running[@]} -gt 0 ] || [ ${#queue[@]} -gt 0 ]; do
  still=()
  for k in "${running[@]}"; do
    s=$(status "$k")
    if [ "$s" = "COMPLETE" ] || [ "$s" = "ERROR" ] || [ "$s" = "CANCEL_ACKNOWLEDGED" ]; then
      mkdir -p "$OUTD/$k"
      $K kernels output "nishkarshsinghal/${SLUG[$k]}" -p "$OUTD/$k" >/dev/null 2>&1
      echo "$(date +%H:%M) $k $s (outputs downloaded)"
      done_list+=("$k")
    else
      still+=("$k")
    fi
  done
  running=("${still[@]}")
  while [ ${#running[@]} -lt 2 ] && [ ${#queue[@]} -gt 0 ]; do
    n=${queue[0]}
    msg=$($K kernels push -p "p6_kernel_$n" 2>&1 | tail -1)
    if echo "$msg" | grep -q "successfully pushed"; then
      echo "$(date +%H:%M) pushed $n"
      running+=("$n"); queue=("${queue[@]:1}")
    else
      break
    fi
  done
  sleep 120
done
echo "$(date +%H:%M) all P6 kernels finished: ${done_list[*]}"

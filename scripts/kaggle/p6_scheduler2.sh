#!/usr/bin/env bash
# Second P6 scheduler: wait for arm A; push the zero-shot kernels (7B first) when a GPU slot frees;
# a push counts as done only if `kernels status` then shows the kernel. Downloads outputs when finished.
K="/c/Users/nishk/anaconda3/python.exe -m kaggle"
cd "D:/Coding/Hackathon/GFG/ARM/kaggle_staging"
OUTD="D:/Coding/Hackathon/GFG/ARM/roadScribe/data/models/p6"
declare -A SLUG=( [A]=roadscribe-p6-arm-a [zs7]=roadscribe-p6-zeroshot-7b [zs]=roadscribe-p6-zeroshot )
declare -A DIR=( [zs7]=p6_kernel_zs7 [zs]=p6_kernel_zs )
running=(A)
queue=(zs7 zs)
status() { $K kernels status "nishkarshsinghal/${SLUG[$1]}" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | sed 's/KernelWorkerStatus\.//'; }
while [ ${#running[@]} -gt 0 ] || [ ${#queue[@]} -gt 0 ]; do
  still=()
  for k in "${running[@]}"; do
    s=$(status "$k")
    if [ "$s" = "COMPLETE" ] || [ "$s" = "ERROR" ] || [ "$s" = "CANCEL_ACKNOWLEDGED" ]; then
      mkdir -p "$OUTD/$k"
      $K kernels output "nishkarshsinghal/${SLUG[$k]}" -p "$OUTD/$k" >/dev/null 2>&1
      echo "$(date +%H:%M) $k $s (outputs downloaded)"
    else
      still+=("$k")
    fi
  done
  running=("${still[@]}")
  if [ ${#queue[@]} -gt 0 ]; then
    n=${queue[0]}
    msg=$($K kernels push -p "${DIR[$n]}" 2>&1 | tr '\n' ' ')
    sleep 20
    s=$(status "$n")
    if [ -n "$s" ] && ! echo "$msg" | grep -q "Maximum batch GPU"; then
      echo "$(date +%H:%M) pushed $n (status $s)"
      running+=("$n"); queue=("${queue[@]:1}")
    else
      echo "$(date +%H:%M) waiting to push $n: ${msg:0:90}"
    fi
  fi
  sleep 120
done
echo "$(date +%H:%M) all done"

#!/bin/bash
# exp 19 launcher: 9 runs, at most 3 concurrently, launch order B=512, 768, 384 (seeds 0,1,2).
# A run is skipped if results/exp19/DROP_B<B>_s<seed> exists at launch time (budget drop, see README).
cd /home/user/ml
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
job() {
  B=$1; S=$2
  [ -e results/exp19/DROP_B${B}_s${S} ] && { echo "dropped B=$B s=$S"; exit 0; }
  python3 -m icl_additive.transformer_exp8 --run free 16 $B $S --out results/exp19/ --max-steps 40000 > results/exp19/logs/free_N16_B${B}_s${S}.log 2>&1
}
export -f job
printf '%s\n' "512 0" "512 1" "512 2" "768 0" "768 1" "768 2" "384 0" "384 1" "384 2" | xargs -P3 -L1 bash -c 'job $0 $1'

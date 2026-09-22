#!/bin/bash
# run_eval.sh <run_name> <cfg_json|null> <k> <seed_base>
#
# AIME26 avg@k for one loop config: 4 vLLM engines x TP=2 on GPUs 0-7, each engine taking
# every 4th sample index (veval.py). About 30 min per run at k=16 on 8x H100.
#
#   VLLM_PY    python of the venv that has vLLM and this plugin (pip install -e vllm_loop)
#   LOOP_RUNS  output root (default /mnt/loop_runs, the local SSD)
#   GPU_PAIRS  subset of the node, e.g. GPU_PAIRS="2,3 4,5 6,7"
set -u
NAME=$1; CFG=$2; K=$3; SEED=$4
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${VLLM_PY:-$HOME/.venvs/vllm/bin/python}
OUT=${LOOP_RUNS:-/mnt/loop_runs}/$NAME
export LOOP_RUNS=${LOOP_RUNS:-/mnt/loop_runs}
export PATH=$(dirname "$PY"):$PATH          # vLLM JIT-compiles with the venv's ninja
# TP>1 fails with "Failed to initialize any NET plugin" unless the NCCL transport is pinned
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
mkdir -p "$OUT"
PAIRS=${GPU_PAIRS:-"0,1 2,3 4,5 6,7"}
NSH=$(echo $PAIRS | wc -w)
pids=""; s=0
for pair in $PAIRS; do
  CUDA_VISIBLE_DEVICES=$pair nohup "$PY" "$HERE/veval.py" "$NAME" "$CFG" "$K" "$SEED" $s $NSH 2 \
    < /dev/null > "$OUT/shard$s.log" 2>&1 &
  pids="$pids $!"; s=$((s+1))
done
echo "$CFG" > "$OUT/cfg.json"
for p in $pids; do wait $p; done
echo "RUN_FINISHED $NAME $(ls "$OUT"/shard*.jsonl 2>/dev/null | wc -l)/$NSH shards"

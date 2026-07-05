#!/bin/bash
# Launch N RWKV7 workers + a front-end router.
#   RWKV7_WORKERS_N  number of workers (default 1; = #NPUs on a multi-NPU box)
#   RWKV7_MODEL, RWKV7_H, RWKV7_N, RWKV7_L, RWKV7_VOCAB  model config
#   RWKV7_SRV_DIR    serving dir (default /root/rwkv7-ascend)
#   RWKV7_PY         python binary (default /usr/local/python3.11.14/bin/python3.11)
# Each worker i is pinned to NPU i via ASCEND_RT_VISIBLE_DEVICES (multi-NPU).
set -e
source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null || source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
SRV=${RWKV7_SRV_DIR:-/root/rwkv7-ascend}
cd "$SRV"
PY=${RWKV7_PY:-/usr/local/python3.11.14/bin/python3.11}
N=${RWKV7_WORKERS_N:-1}
MODEL=${RWKV7_MODEL:-$SRV/models/rwkv7-g1g-1.5b-hf}
H=${RWKV7_H:-32}; NN=${RWKV7_N:-64}; L=${RWKV7_L:-24}
VOCAB=${RWKV7_VOCAB:-$SRV/assets/rwkv_vocab_v20230424.txt}
URLS=""
for i in $(seq 0 $((N-1))); do
  PORT=$((8001+i))
  setsid bash -c "ASCEND_RT_VISIBLE_DEVICES=$i RWKV7_REQ_TIMEOUT=120 $PY serve_full.py --model $MODEL --H $H --N $NN --L $L --vocab $VOCAB --port $PORT" </dev/null >/tmp/rwkv7_worker_$i.log 2>&1 &
  URLS="$URLS,http://localhost:$PORT"
  echo "[cluster] worker $i -> port $PORT, NPU $i (pid $!)"
done
URLS=${URLS#,}
sleep 2
RWKV7_WORKERS=$URLS setsid $PY serve_router.py --port 8000 </dev/null >/tmp/rwkv7_router.log 2>&1 &
echo "[cluster] router -> port 8000, workers=[$URLS] (pid $!)"

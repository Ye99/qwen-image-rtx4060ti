#!/bin/bash
# Start the Qwen-Image-2.1 LightX2V server on both RTX 4060 Ti cards (FP8, Ulysses SP + text-encoder TP).
# Compiles once (~60-120 s), then serves requests; pair with prompt_loop.py. Usage: bash serve_4060ti.sh
# (PORT=8000, HOST=127.0.0.1 by default; Ctrl-C stops it within a few seconds, even mid-generation)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
config=$(render_config qwen_image_21_4060ti_1k_sp2.json)

# Ctrl-C handling: torchrun's own SIGINT path hangs ~30 s when a request is in flight (rank 1 exits at
# once, rank 0's uvicorn waits on a request stuck in an NCCL collective), then dumps a traceback.
# torchrun starts each worker in its own session, so signal workers by PID, not process group:
# cancel running tasks (aborts the denoising loop), drop torchrun quietly, SIGTERM the workers,
# and SIGKILL any still alive after STOP_TIMEOUT seconds. Nothing is stateful.
setsid torchrun --nproc_per_node=2 -m lightx2v.server \
    --model_cls qwen_image_21 \
    --model_path "${model_path}" \
    --config_json "${config}" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-8000}" &
torchrun_pid=$!

any_alive() {
    local p
    for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done
    return 1
}

shutdown() {
    trap - INT TERM
    echo "[serve] stopping..." >&2
    local workers
    workers=$(pgrep -P "$torchrun_pid" | tr '\n' ' ')
    curl -s -m 3 -o /dev/null -X DELETE "http://127.0.0.1:${PORT:-8000}/v1/tasks/all/running" || true
    kill -KILL "$torchrun_pid" 2>/dev/null || true
    [ -n "$workers" ] && kill -TERM $workers 2>/dev/null
    local deadline=$((SECONDS + ${STOP_TIMEOUT:-10}))
    while any_alive $workers && [ $SECONDS -lt $deadline ]; do sleep 0.2; done
    if any_alive $workers; then
        echo "[serve] workers still busy, force-stopping" >&2
        kill -KILL $workers 2>/dev/null || true
    fi
    wait "$torchrun_pid" 2>/dev/null || true
    rm -f "$config"
    echo "[serve] stopped" >&2
    exit 0
}
trap shutdown INT TERM
wait "$torchrun_pid"

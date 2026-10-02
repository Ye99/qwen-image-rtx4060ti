#!/bin/bash
# One-shot Qwen-Image-2.1 T2I with LightX2V FP8 on RTX 4060 Ti 16 GB (compiles every run; use serve_4060ti.sh for many images).
# Usage: NGPU=1|2 PROMPT="..." SEED=42 WIDTH=1024 HEIGHT=1024 bash run_4060ti.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NGPU=${NGPU:-1}
if [ "$NGPU" = 2 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    launch=(torchrun --nproc_per_node=2 -m lightx2v.infer); config=$(render_config qwen_image_21_4060ti_1k_sp2.json)
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    launch=(python -m lightx2v.infer); config=$(render_config qwen_image_21_4060ti_1k.json)
fi
trap 'rm -f "$config"' EXIT
"${launch[@]}" \
    --model_cls qwen_image_21 --task t2i \
    --model_path "${model_path}" \
    --config_json "${config}" \
    --prompt "${PROMPT:-A capybara wearing a wizard hat sits at a desk, reading a book by candlelight. Oil painting style, delicate brushwork, warm tones.}" \
    --size "${HEIGHT:-1024}" "${WIDTH:-1024}" \
    --seed "${SEED:-42}" \
    --save_result_path "${SAVE_RESULT_PATH:-${REPO_DIR}/lightx2v/save_results/t2i_4060ti_${NGPU}gpu.png}"

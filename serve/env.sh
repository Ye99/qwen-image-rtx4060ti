# Shared setup for run_4060ti.sh / serve_4060ti.sh. Sourced, not executed.
# Default layout (override any of these with environment variables):
#   $WORK_DIR/LightX2V                 LightX2V clone with its .venv          (LIGHTX2V_PATH)
#   $WORK_DIR/models                   FP8 checkpoints from tools/convert     (FP8_DIR)
#   $WORK_DIR/qwen-image-rtx4060ti     this repo
#   HF cache snapshot of Qwen/Qwen-Image-2.1                                  (MODEL_PATH)

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=${WORK_DIR:-$(dirname "$REPO_DIR")}
lightx2v_path=${LIGHTX2V_PATH:-$WORK_DIR/LightX2V}
FP8_DIR=${FP8_DIR:-$WORK_DIR/models}
model_path=${MODEL_PATH:-$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--Qwen--Qwen-Image-2.1/snapshots/* 2>/dev/null | head -1)}
[ -d "$model_path" ] || { echo "Qwen-Image-2.1 weights not found; run: hf download Qwen/Qwen-Image-2.1 (or set MODEL_PATH)" >&2; exit 1; }

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export PATH=${lightx2v_path}/.venv/bin:${CUDA_HOME}/bin:$PATH
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
source "${lightx2v_path}/scripts/base/base.sh"

# Configs reference checkpoints as __FP8_DIR__/...; resolve them into a temp copy.
render_config() {
    local out
    out=$(mktemp --suffix=.json)
    sed "s#__FP8_DIR__#${FP8_DIR}#g" "$REPO_DIR/serve/configs/$1" > "$out"
    echo "$out"
}

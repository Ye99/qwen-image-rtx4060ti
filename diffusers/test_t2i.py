import time

import torch
from diffusers import QwenImage21Pipeline
from diffusers.hooks import apply_group_offloading

# Same prompt/seed/size/steps as LightX2V's qwen_image_21_t2i_5090_1k.sh, for a like-for-like comparison.
PROMPT = "A capybara wearing a wizard hat sits at a desk, reading a book by candlelight. Oil painting style, delicate brushwork, warm tones."

t = time.time()
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16)
# bf16 text encoder alone (16.3 GiB) exceeds a 16 GB card, so enable_model_cpu_offload() OOMs.
# Stream weights to the GPU on demand instead, prefetching on a side CUDA stream. Block-level grouping
# never releases the transformers text encoder's blocks (OOM mid-encode), so that one goes leaf by leaf.
cuda = torch.device("cuda")
pipe.transformer.enable_group_offload(cuda, offload_type="block_level", num_blocks_per_group=1, use_stream=True)
apply_group_offloading(pipe.text_encoder, cuda, offload_type="leaf_level", use_stream=True)
pipe.vae.to(cuda)
print(f"load {time.time() - t:.1f}s")


def generate():
    return pipe(
        prompt=PROMPT,
        width=1024,
        height=1024,
        num_inference_steps=40,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]


for run in ("warmup", "timed"):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    image = generate()
    print(f"{run}: {time.time() - t:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
image.save("test_t2i.png")

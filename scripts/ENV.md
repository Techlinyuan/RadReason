# Environment notes (dev machine)

Tested config that runs the full pipeline single-GPU, no vLLM/DeepSpeed:

| component | version |
|---|---|
| python | 3.10 |
| torch | 2.6.0 + cu124 |
| transformers | 4.57.6 (native `Qwen3VLForConditionalGeneration`) |
| peft | 0.19.1 |
| datasets | 4.8.5 |
| GPU | 1× (24GB+ recommended; 4B + LoRA + G=8 rollouts) |

## Restricted network (HF/GitHub blocked)
```
set HF_ENDPOINT=https://hf-mirror.com   REM datasets & model hub via mirror
```

## Pick a free GPU
```
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
set CUDA_VISIBLE_DEVICES=<free_id>       REM the code always uses cuda:0 = the visible card
```

## Memory tips (if OOM)
- lower `--group` (e.g. 4) and `--max-new-tokens` (e.g. 192)
- lower `--max-side` (e.g. 384) to shrink the vision tokens
- keep `prompts-per-step` at 1

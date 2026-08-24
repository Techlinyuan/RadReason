@echo off
cd /d G:\XiaoqianZhu\jianli
set CUDA_VISIBLE_DEVICES=1
set HF_ENDPOINT=https://hf-mirror.com
G:\ProgramData\anaconda3\envs\vlmgym\python.exe -m src.radreason.eval --data data/test_slake.jsonl --model G:\XiaoqianZhu\models\Qwen\Qwen3-VL-4B-Instruct --adapter outputs/grpo_v2/lora_adapter --out results/grpo_v2_slake.json --samples-md results/samples_grpo_v2_slake.md --limit 400 --max-new-tokens 384 > G:\XiaoqianZhu\jianli\outputs\eval_v2_slake.log 2>&1

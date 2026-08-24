@echo off
cd /d G:\XiaoqianZhu\jianli
set CUDA_VISIBLE_DEVICES=0
set HF_ENDPOINT=https://hf-mirror.com
set WANDB_MODE=offline
G:\ProgramData\anaconda3\envs\vlmgym\python.exe -m src.radreason.grpo_train --data data/train.jsonl --model G:\XiaoqianZhu\models\Qwen\Qwen3-VL-4B-Instruct --out outputs/grpo_v2 --steps 200 --group 8 --prompts-per-step 1 --dynamic-sampling --max-prompt-tries 6 --max-new-tokens 384 --save-every 20 --wandb > G:\XiaoqianZhu\jianli\outputs\v2_train.log 2>&1

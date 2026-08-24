@echo off
REM ===== RadReason GRPO training (Windows / single GPU) =====
REM Usage: scripts\run_grpo.bat [GPU_ID]
setlocal
set REPO=%~dp0..
set PY=G:\ProgramData\anaconda3\envs\vlmgym\python.exe
set MODEL=G:\XiaoqianZhu\models\Qwen\Qwen3-VL-4B-Instruct

set GPU=%1
if "%GPU%"=="" set GPU=0
set CUDA_VISIBLE_DEVICES=%GPU%
set HF_ENDPOINT=https://hf-mirror.com
set WANDB_MODE=offline

cd /d %REPO%
"%PY%" -m src.radreason.grpo_train ^
  --data data/train.jsonl ^
  --model "%MODEL%" ^
  --out outputs/grpo ^
  --steps 300 --group 8 --prompts-per-step 2 --lr 2e-5 ^
  --max-new-tokens 256 --temperature 1.1 --top-p 0.95 --wandb
endlocal

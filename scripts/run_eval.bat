@echo off
REM ===== RadReason evaluation: zero-shot vs GRPO-tuned =====
REM Usage: scripts\run_eval.bat [GPU_ID]
setlocal
set REPO=%~dp0..
set PY=G:\ProgramData\anaconda3\envs\vlmgym\python.exe
set MODEL=G:\XiaoqianZhu\models\Qwen\Qwen3-VL-4B-Instruct

set GPU=%1
if "%GPU%"=="" set GPU=0
set CUDA_VISIBLE_DEVICES=%GPU%
set HF_ENDPOINT=https://hf-mirror.com

cd /d %REPO%
echo [1/2] zero-shot baseline...
"%PY%" -m src.radreason.eval --data data/test.jsonl --model "%MODEL%" ^
  --out results/base.json --samples-md results/samples_base.md --limit 400

echo [2/2] GRPO-tuned...
"%PY%" -m src.radreason.eval --data data/test.jsonl --model "%MODEL%" ^
  --adapter outputs/grpo/lora_adapter ^
  --out results/grpo.json --samples-md results/samples_grpo.md --limit 400
endlocal

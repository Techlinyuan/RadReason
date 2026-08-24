# -*- coding: utf-8 -*-
"""Wait for a free GPU on the shared server, then launch GRPO training on it.
Polite: only grabs a card with enough free memory and low utilization.
"""
import subprocess, time, os

REPO = r"G:\XiaoqianZhu\jianli"
PY = r"G:\ProgramData\anaconda3\envs\vlmgym\python.exe"
MODEL = r"G:\XiaoqianZhu\models\Qwen\Qwen3-VL-4B-Instruct"
MIN_FREE_MB = 26000
MAX_UTIL = 50


def pick_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"]).decode()
    except Exception as e:
        print("nvidia-smi err:", e, flush=True); return None
    for line in out.strip().splitlines():
        i, free, util = [x.strip() for x in line.split(",")]
        if float(free) >= MIN_FREE_MB and float(util) <= MAX_UTIL:
            return i
    return None


print("waiting for a free GPU (>=26GB free, util<=50%) ...", flush=True)
waited = 0
while True:
    g = pick_gpu()
    if g is not None:
        # confirm it stays free for a few seconds (avoid transient gaps)
        time.sleep(6)
        if pick_gpu() == g:
            print(f"acquired GPU {g} after {waited}s", flush=True)
            break
    time.sleep(30); waited += 30
    if waited % 300 == 0:
        print(f"  still waiting ({waited}s) ...", flush=True)

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = g
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["WANDB_MODE"] = "offline"
cmd = [PY, "-m", "src.radreason.grpo_train",
       "--data", "data/train.jsonl", "--model", MODEL, "--out", "outputs/grpo",
       "--steps", "200", "--group", "8", "--prompts-per-step", "1",
       "--lr", "2e-5", "--max-new-tokens", "384", "--save-every", "40", "--wandb"]
print("LAUNCH:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.run(cmd, env=env, cwd=REPO).returncode)

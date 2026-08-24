# -*- coding: utf-8 -*-
"""FastAPI server: serve the GRPO-tuned reasoning VLM as a /diagnose endpoint.

  POST /diagnose  (multipart: image=<file>, question=<str>)
    -> {"reasoning": "...", "answer": "...", "raw": "..."}

Env:
  MODEL    path to Qwen3-VL-4B-Instruct              (required)
  ADAPTER  path to the trained LoRA adapter dir       (optional; omit = zero-shot base)
  LOAD_4BIT=1   load base in 4-bit (bitsandbytes) for lower memory / faster inference
"""
import os, io
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from radreason.modeling import build_inputs, generate_text          # noqa: E402
from radreason.rewards import extract_answer                        # noqa: E402
from radreason.prompts import build_messages                        # noqa: E402
import re

app = FastAPI(title="RadReason", version="0.1.0")
_MODEL = _PROC = None


def _load():
    global _MODEL, _PROC
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    path = os.environ["MODEL"]
    _PROC = AutoProcessor.from_pretrained(path)
    kw = dict(dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if os.environ.get("LOAD_4BIT") == "1":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    m = Qwen3VLForConditionalGeneration.from_pretrained(path, **kw)
    if os.environ.get("ADAPTER"):
        m = PeftModel.from_pretrained(m, os.environ["ADAPTER"])
    m.eval()
    _MODEL = m


@app.on_event("startup")
def _startup():
    _load()


@app.get("/health")
def health():
    return {"status": "ok", "adapter": os.environ.get("ADAPTER", None)}


@app.post("/diagnose")
async def diagnose(image: UploadFile = File(...), question: str = Form(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    inp = build_inputs(_PROC, img, question, "cuda:0")
    _, text = generate_text(_MODEL, _PROC, inp, max_new_tokens=256, do_sample=False)
    think = re.search(r"<think>(.*?)</think>", text, re.S | re.I)
    return {"reasoning": think.group(1).strip() if think else "",
            "answer": extract_answer(text), "raw": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

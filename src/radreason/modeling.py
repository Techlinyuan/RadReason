# -*- coding: utf-8 -*-
"""Qwen3-VL loading + the three primitives GRPO needs:
   build_inputs (image+text -> model inputs), generate_text (rollout), seq_mean_logprob.

The generate / logprob mechanics mirror a verified single-GPU Qwen3-VL setup
(no vLLM / FSDP / NCCL); rollouts use HuggingFace `generate`.
"""
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel

from .prompts import build_messages


def load_model(model_path, gpu=0, dtype=torch.bfloat16, attn="sdpa",
               lora=None, adapter=None, train=True, trainable=False):
    """Load base Qwen3-VL; attach a fresh LoRA (train), or an existing adapter.

    `trainable=True` keeps the loaded adapter's weights differentiable, which is what
    resuming an interrupted GRPO run needs; the default (False) is frozen, for eval.
    """
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map={"": gpu}, attn_implementation=attn)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        if trainable:
            model.print_trainable_parameters()
    elif lora:
        cfg = LoraConfig(r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32),
                         lora_dropout=lora.get("dropout", 0.0), bias="none",
                         target_modules=lora.get("target_modules",
                                                 ["q_proj", "k_proj", "v_proj", "o_proj"]),
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    model.config.use_cache = True
    if not train:
        model.eval()
    return model, processor


def _maybe_downscale(img, max_side=512):
    img = img.convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
    return img


def build_inputs(processor, image, question, device, max_side=512):
    if isinstance(image, str):
        image = Image.open(image)
    image = _maybe_downscale(image, max_side)
    messages = build_messages(image, question)
    inp = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    inp.pop("token_type_ids", None)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}


@torch.no_grad()
def generate_text(model, processor, inp, max_new_tokens=256,
                  do_sample=True, temperature=1.1, top_p=0.95):
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=do_sample,
                         temperature=temperature, top_p=top_p)
    gen = out[0, inp["input_ids"].shape[1]:]
    text = processor.decode(gen, skip_special_tokens=True)
    return gen, text


def seq_mean_logprob(model, inp, gen_ids):
    """Mean log-prob of the generated tokens under the current policy (with grad)."""
    P, T = inp["input_ids"].shape[1], gen_ids.shape[0]
    full = torch.cat([inp["input_ids"], gen_ids.view(1, T)], dim=1)
    kw = dict(input_ids=full, attention_mask=torch.ones_like(full))
    for k in ("pixel_values", "image_grid_thw"):
        if k in inp:
            kw[k] = inp[k]
    logits = model(**kw).logits[:, P - 1:P + T - 1, :]
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, gen_ids.view(1, T, 1))
    return lp.squeeze(-1).mean()

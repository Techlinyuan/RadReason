# -*- coding: utf-8 -*-
"""Verifiable rewards for GRPO: answer correctness + output-format compliance.

No reward model, no labeled rationales — the reward is computable from the gold answer,
which is exactly what makes closed-form medical VQA a good fit for GRPO.
"""
import re

_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_PUNCT = re.compile(r"[\s\.\,\;\:\!\?\"'`\(\)\[\]\{\}]+")


def extract_answer(text):
    """Return the content of the last <answer>...</answer>, else a best-effort fallback."""
    m = list(_ANSWER.finditer(text or ""))
    if m:
        return m[-1].group(1).strip()
    # fallback: after 'answer:' or last line
    m2 = re.search(r"answer\s*[:：]\s*(.+)", text or "", re.I)
    if m2:
        return m2.group(1).strip()
    return (text or "").strip().splitlines()[-1].strip() if (text or "").strip() else ""


def normalize(s):
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s).strip()
    # common yes/no canonicalization
    if s in {"yes", "yeah", "yep", "true", "present"}:
        return "yes"
    if s in {"no", "nope", "false", "absent", "none"}:
        return "no"
    return s


def format_score(text, bonus=0.5):
    """Reward well-formed <think>...</think><answer>...</answer> (partial credit)."""
    has_think = bool(_THINK.search(text or ""))
    has_answer = bool(_ANSWER.search(text or ""))
    if has_think and has_answer:
        return bonus
    if has_answer:
        return 0.5 * bonus
    return 0.0


def accuracy_score(text, gold, correct=1.0, wrong=0.0):
    """Exact-match (normalized) accuracy on the extracted <answer>."""
    pred = normalize(extract_answer(text))
    g = normalize(gold)
    if not pred:
        return wrong, 0
    if pred == g:
        return correct, 1
    # for multi-word gold, accept exact set match / containment of the gold phrase
    if len(g.split()) >= 2 and g in pred:
        return correct, 1
    return wrong, 0


def compute_reward(text, gold, cfg=None):
    """Total scalar reward = accuracy + format bonus. Returns (reward, is_correct)."""
    cfg = cfg or {}
    acc, is_correct = accuracy_score(
        text, gold, cfg.get("correct", 1.0), cfg.get("wrong", 0.0))
    fmt = format_score(text, cfg.get("format_bonus", 0.5))
    return acc + fmt, is_correct

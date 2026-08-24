# -*- coding: utf-8 -*-
import os, random, json
import numpy as np


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_closed(answer):
    """Heuristic: yes/no or a short (<=2 word) answer counts as closed / verifiable."""
    a = (answer or "").strip().lower()
    if a in {"yes", "no"}:
        return True
    return 0 < len(a.split()) <= 2

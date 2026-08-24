# -*- coding: utf-8 -*-
"""Evaluate zero-shot vs GRPO-tuned: greedy decode, parse <answer>, exact-match accuracy.

Reports overall + per-dataset + closed/open accuracy, and dumps qualitative samples.
"""
import os, json, argparse
_v = os.environ.get("HF_ENDPOINT", "").strip()
os.environ["HF_ENDPOINT"] = _v if _v else "https://hf-mirror.com"  # strip trailing-space bug
from collections import defaultdict

from .utils import read_jsonl
from .rewards import accuracy_score, extract_answer
from .modeling import load_model, build_inputs, generate_text


def evaluate(args):
    dev = "cuda:0"
    data = read_jsonl(args.data)
    if args.limit:
        data = data[:args.limit]
    data_dir = os.path.dirname(os.path.abspath(args.data))
    model, processor = load_model(args.model, gpu=0, adapter=args.adapter, train=False)

    tot = defaultdict(int); cor = defaultdict(int); samples = []
    for i, ex in enumerate(data):
        img_path = ex["image"] if os.path.isabs(ex["image"]) else os.path.join(data_dir, ex["image"])
        inp = build_inputs(processor, img_path, ex["question"], dev, args.max_side)
        _, text = generate_text(model, processor, inp, max_new_tokens=args.max_new_tokens,
                                do_sample=False)
        acc, ok = accuracy_score(text, ex["answer"])
        ds, qt = ex.get("dataset", "?"), ex.get("qtype", "?")
        for key in ["all", f"ds:{ds}", f"qt:{qt}"]:
            tot[key] += 1; cor[key] += ok
        if len(samples) < args.n_samples:
            samples.append(dict(dataset=ds, question=ex["question"], gold=ex["answer"],
                                pred=extract_answer(text), correct=bool(ok), raw=text))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(data)}  acc(all)={cor['all']/max(1,tot['all']):.3f}", flush=True)

    report = {k: dict(n=tot[k], acc=round(cor[k] / max(1, tot[k]), 4)) for k in sorted(tot)}
    out = dict(model=args.model, adapter=args.adapter, n=len(data), report=report)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if args.samples_md:
        _dump_samples(samples, args.samples_md, args.adapter)
    print("=== RESULT ===")
    for k, v in report.items():
        print(f"  {k:16s} n={v['n']:4d}  acc={v['acc']:.4f}")
    print("saved ->", args.out)
    return out


def _dump_samples(samples, path, adapter):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tag = "GRPO-tuned" if adapter else "zero-shot"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Qualitative samples ({tag})\n\n")
        for s in samples:
            f.write(f"**[{s['dataset']}]** Q: {s['question']}\n\n")
            f.write(f"- gold: `{s['gold']}` · pred: `{s['pred']}` · "
                    f"{'✅' if s['correct'] else '❌'}\n\n")
            f.write(f"```\n{s['raw'].strip()[:800]}\n```\n\n---\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for zero-shot)")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--samples-md", default="results/samples.md")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-side", type=int, default=512)
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()

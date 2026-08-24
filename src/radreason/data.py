# -*- coding: utf-8 -*-
"""Download public radiology VQA benchmarks and unify them to one schema.

Output:
  <out>/images/*.png
  <out>/train.jsonl , <out>/test.jsonl   with rows:
      {"image": "images/xxx.png", "question": str, "answer": str,
       "dataset": "vqa-rad"|"slake", "split": "train"|"test", "qtype": "closed"|"open"}
"""
import os, argparse
from .utils import write_jsonl, is_closed

# HF dataset ids (all share {image: PIL, question, answer})
SOURCES = {
    "vqa-rad": "flaviagiammarino/vqa-rad",
    "slake":  "mdwiratathya/SLAKE-vqa-english",
}


def _load_split(hf_id, split):
    from datasets import load_dataset
    try:
        return load_dataset(hf_id, split=split)
    except Exception:
        return None


def _clean_hf_endpoint():
    # guard against the classic 'set VAR=val & cmd' trailing-space -> hf-mirror.com%20 bug
    v = os.environ.get("HF_ENDPOINT", "").strip()
    os.environ["HF_ENDPOINT"] = v if v else "https://hf-mirror.com"


def prepare(datasets, out_dir, closed_only_for_train=True,
            max_train=2000, max_eval=800, seed=0):
    _clean_hf_endpoint()
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    train_rows, test_rows = [], []

    for name in datasets:
        hf_id = SOURCES[name]
        for split, bucket, cap in [("train", train_rows, max_train),
                                   ("test", test_rows, max_eval)]:
            ds = _load_split(hf_id, split)
            if ds is None:
                print(f"[{name}] split '{split}' not found, skipping")
                continue
            ds = ds.shuffle(seed=seed)
            n = 0
            for i, ex in enumerate(ds):
                q = (ex.get("question") or "").strip()
                a = (ex.get("answer") or "").strip()
                img = ex.get("image")
                if not q or not a or img is None:
                    continue
                qtype = "closed" if is_closed(a) else "open"
                if split == "train" and closed_only_for_train and qtype != "closed":
                    continue
                rel = f"images/{name}_{split}_{i}.png"
                try:
                    img.convert("RGB").save(os.path.join(out_dir, rel))
                except Exception:
                    continue
                bucket.append({"image": rel, "question": q, "answer": a,
                               "dataset": name, "split": split, "qtype": qtype})
                n += 1
                if n >= cap:
                    break
            print(f"[{name}] {split}: kept {n}")

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test_rows)
    print(f"TOTAL train={len(train_rows)}  test={len(test_rows)}  -> {out_dir}")
    return train_rows, test_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["vqa-rad", "slake"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-eval", type=int, default=800)
    ap.add_argument("--all-qtypes-train", action="store_true",
                    help="keep open questions in train too (default: closed only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prepare(a.datasets, a.out, closed_only_for_train=not a.all_qtypes_train,
            max_train=a.max_train, max_eval=a.max_eval, seed=a.seed)


if __name__ == "__main__":
    main()

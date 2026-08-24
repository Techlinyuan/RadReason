# -*- coding: utf-8 -*-
"""GRPO training for medical VQA reasoning.

For each (image, question): sample G completions, score each with a verifiable reward,
form group-relative advantages, and take a policy-gradient step on the LoRA params.
Single GPU, no vLLM / FSDP. Mirrors a verified Qwen3-VL GRPO loop.
"""
import os, json, time, random, argparse
_v = os.environ.get("HF_ENDPOINT", "").strip()
os.environ["HF_ENDPOINT"] = _v if _v else "https://hf-mirror.com"  # strip trailing-space bug
import numpy as np
import torch

from .utils import set_seed, read_jsonl
from .rewards import compute_reward
from .modeling import load_model, build_inputs, generate_text, seq_mean_logprob


def _save_state(out, model, opt, hist, step):
    """Adapter + optimizer moments + step/history, so an interrupted run can resume."""
    model.save_pretrained(os.path.join(out, "lora_adapter"))
    torch.save(opt.state_dict(), os.path.join(out, "optimizer.pt"))
    json.dump({"step": step, "hist": hist},
              open(os.path.join(out, "train_state.json"), "w"), indent=2)


def _restore_state(resume_dir, opt, dev):
    """Read (start_step, hist) and optimizer moments saved next to the adapter."""
    parent = os.path.dirname(os.path.normpath(os.path.abspath(resume_dir)))
    start, hist = 0, []
    sp = os.path.join(parent, "train_state.json")
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        start, hist = int(d.get("step", 0)), d.get("hist", [])
        print(f"restored {len(hist)} logged steps; continuing after step {start}", flush=True)
    else:
        print("no train_state.json next to the adapter: weights resume, step counter restarts",
              flush=True)
    op = os.path.join(parent, "optimizer.pt")
    if os.path.exists(op):
        try:
            opt.load_state_dict(torch.load(op, map_location=dev))
            print("restored optimizer state", flush=True)
        except Exception as e:
            print(f"optimizer state not restored ({type(e).__name__}); fresh moments", flush=True)
    return start, hist


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES selects the physical card
    data = read_jsonl(args.data)
    data_dir = os.path.dirname(os.path.abspath(args.data))
    print(f"loaded {len(data)} training QA from {args.data}", flush=True)

    lora = dict(r=16, alpha=32, dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    if args.resume:
        print(f"resuming from adapter: {args.resume}", flush=True)
        model, processor = load_model(args.model, gpu=0, adapter=args.resume,
                                      trainable=True, train=True)
    else:
        model, processor = load_model(args.model, gpu=0, lora=lora, train=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    start_step, hist = _restore_state(args.resume, opt, dev) if args.resume else (0, [])
    if start_step:
        set_seed(args.seed + start_step)   # avoid replaying the exact prompts already trained on
    rcfg = dict(format_bonus=args.format_bonus, correct=1.0, wrong=0.0)

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project="radreason", name="qwen3vl4b_grpo",
                            mode=os.environ.get("WANDB_MODE", "offline"),
                            config=vars(args))
        except Exception as e:
            print("wandb off:", repr(e)); wb = None

    print(f"=== RadReason GRPO: steps {start_step + 1}..{args.steps} "
          f"prompts/step={args.prompts_per_step} G={args.group} lr={args.lr} "
          f"dynamic_sampling={args.dynamic_sampling} ===", flush=True)
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time(); comp = []; skipped = 0
        got, tries = 0, 0
        max_tries = args.max_prompt_tries if args.dynamic_sampling else args.prompts_per_step
        while got < args.prompts_per_step and tries < max_tries:
            tries += 1
            ex = random.choice(data)
            img_path = ex["image"] if os.path.isabs(ex["image"]) else os.path.join(data_dir, ex["image"])
            inp = build_inputs(processor, img_path, ex["question"], dev, args.max_side)
            items, rs = [], []
            for _ in range(args.group):
                gen, text = generate_text(model, processor, inp,
                                          max_new_tokens=args.max_new_tokens,
                                          temperature=args.temperature, top_p=args.top_p)
                r, is_correct = compute_reward(text, ex["answer"], rcfg)
                rs.append(r)
                items.append(dict(inp=inp, gen=gen, r=r, correct=is_correct,
                                  tlen=int(gen.shape[0])))
            rs = np.array(rs, dtype=np.float32)
            if args.dynamic_sampling and rs.std() < 1e-6:
                # DAPO-style dynamic sampling: zero within-group reward variance means
                # zero advantage for every sample (no learning signal) -> resample a new prompt
                skipped += 1
                continue
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)   # group-relative advantage
            for j, it in enumerate(items):
                it["adv"] = float(adv[j]); comp.append(it)
            got += 1

        opt.zero_grad(); losses = []
        for it in comp:
            if abs(it["adv"]) < 1e-8:
                continue
            loss = -(it["adv"] * seq_mean_logprob(model, it["inp"], it["gen"])) / max(1, len(comp))
            loss.backward(); losses.append(float(loss.detach()))
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        opt.step()

        allr = np.array([it["r"] for it in comp]) if comp else np.array([0.0])
        m = dict(step=step, pg_loss=float(np.sum(losses)) if losses else 0.0,
                 reward_mean=float(allr.mean()), reward_std=float(allr.std()),
                 acc=float(np.mean([it["correct"] for it in comp])) if comp else 0.0,
                 resp_len=float(np.mean([it["tlen"] for it in comp])) if comp else 0.0,
                 skipped=skipped, groups=got,
                 grad_norm=float(gn), sec=time.time() - t0)
        hist.append(m)
        print("step %3d | loss %+.4f | reward %+.3f | acc %.2f | skip %d | gnorm %.2f | %.1fs"
              % (step, m["pg_loss"], m["reward_mean"], m["acc"], m["skipped"], m["grad_norm"], m["sec"]),
              flush=True)
        if wb:
            wb.log({k: v for k, v in m.items() if k != "step"}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            _save_state(args.out, model, opt, hist, step)

    _save_state(args.out, model, opt, hist, args.steps)
    json.dump(hist, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    _plot(hist, args.out)
    if wb:
        try: wb.finish()
        except Exception: pass
    print("SAVED adapter ->", os.path.join(args.out, "lora_adapter"), flush=True)
    print("ALL_DONE", flush=True)


def _plot(hist, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [h["step"] for h in hist]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(xs, [h["reward_mean"] for h in hist], "-o", ms=3); ax[0].set_title("reward_mean")
        ax[1].plot(xs, [h["acc"] for h in hist], "-o", ms=3, color="C2"); ax[1].set_title("train_accuracy")
        ax[2].plot(xs, [h["pg_loss"] for h in hist], "-o", ms=3, color="C3"); ax[2].set_title("pg_loss")
        for a in ax: a.grid(alpha=.3); a.set_xlabel("step")
        plt.tight_layout(); plt.savefig(os.path.join(out, "curves.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="outputs/grpo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--format-bonus", type=float, default=0.5)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dynamic-sampling", action="store_true",
                    help="DAPO-style: resample prompts whose G rollouts all get the same reward")
    ap.add_argument("--max-prompt-tries", type=int, default=10,
                    help="max prompts sampled per step when dynamic sampling is on")
    ap.add_argument("--resume", default=None,
                    help="adapter dir of an interrupted run (e.g. outputs/grpo_v2/lora_adapter); "
                         "--steps stays the TOTAL step count, not extra steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()

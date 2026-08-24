# Method

## Problem
Medical visual question answering (VQA) with a vision-language model (VLM). We want the model to
**reason before answering** and to be **more accurate** than the zero-shot base model — without any
human-written reasoning traces.

## Why GRPO (and not SFT)
Supervised fine-tuning would require gold reasoning chains, which we do not have. **GRPO** (Group Relative
Policy Optimization) only needs a **scalar reward per sampled answer**. For closed-form medical VQA the reward
is *verifiable* (the gold answer is known), so we can reward correctness directly and let the reasoning emerge.

GRPO is **critic-free**: instead of a learned value function, it normalizes rewards *within a group* of `G`
samples drawn for the same prompt.

## Training objective
For a prompt `x` (image + question) we sample `G` completions `y₁..y_G ~ πθ(·|x)`.
Each gets reward `rᵢ`. The group-relative advantage is

```
Âᵢ = (rᵢ − mean(r)) / (std(r) + ε)
```

and we maximize `Σᵢ Âᵢ · meanₜ log πθ(y_{i,t} | x, y_{i,<t})`, i.e. minimize

```
L(θ) = − (1/N) Σᵢ Âᵢ · meanₜ log πθ(tokenₜ)
```

Only the **LoRA** parameters (q/k/v/o projections) are trained; the base VLM is frozen.

## Reward
```
r(y) = accuracy(answer(y), gold)          # 1 if normalized exact-match else 0
     + format_bonus · 1[well-formed <think>…</think><answer>…</answer>]
```
- `answer(y)` = content of the last `<answer>…</answer>`.
- normalization: lowercase, strip punctuation, canonicalize yes/no.
- The format bonus stabilizes early training (before accuracy signal is dense).

## Rollout
Completions are produced by HuggingFace `model.generate` with sampling (`temperature`, `top_p`) — **no vLLM**.
This is slower than an inference-engine rollout but has zero extra dependencies and runs on Windows / a single GPU.

## Data
- **SLAKE** (English) and **VQA-RAD** — public radiology VQA. Unified to `{image, question, answer}`.
- Training uses **closed-form** questions (verifiable). Evaluation reports closed-set exact-match accuracy on
  held-out test splits (in-domain SLAKE and cross-dataset VQA-RAD).

## Evaluation
Greedy decoding, parse `<answer>`, exact-match accuracy — overall, per-dataset, and closed/open breakdown.
We compare **zero-shot** vs **GRPO-tuned** and log qualitative reasoning samples.

## Limitations (honest)
- Reward is exact-match on short answers → best suited to closed VQA; open-ended generation is out of scope.
- HF-`generate` rollouts make GRPO slower than vLLM-based pipelines.
- Small base model (4B) + LoRA: gains are in *reasoning behavior + accuracy on closed VQA*, not SOTA on every benchmark.

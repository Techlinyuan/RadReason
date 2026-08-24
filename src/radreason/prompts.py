# -*- coding: utf-8 -*-
"""R1-style prompt templates: force <think> reasoning </think> then <answer> a </answer>."""

SYSTEM_PROMPT = (
    "You are an expert radiologist assisting with medical image interpretation. "
    "Examine the image and answer the question.\n"
    "You MUST respond in exactly this format and nothing else:\n"
    "<think> brief reasoning, at most 2 short sentences </think>\n"
    "<answer> the final answer in a few words </answer>\n"
    "Always close both tags. Keep the reasoning short so you do not run out of space "
    "before writing the <answer>."
)

USER_TEMPLATE = (
    "Question: {question}\n"
    "Give brief reasoning inside <think></think>, then the short final answer inside <answer></answer>."
)


def build_messages(image, question):
    """Chat messages for Qwen3-VL processor.apply_chat_template."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_TEMPLATE.format(question=question)},
        ]},
    ]

"""MOT-6: a minimal behavioral cross-check. Every other measurement in this
project is representational/geometric (do internal vectors converge? does
the logit lens detour?) -- none of it directly tests whether any of this
predicts real Urdu-vs-Hindi behavioral differences. This asks a much more
direct question on a small factual QA set (data/qa/factual_qa.json, ~30
items with known short answers): does the model actually answer correctly in
Urdu vs. Hindi? Deliberately not elaborate (RESOLUTIONS.md: "it needs to
exist"), just real and scored automatically (exact/near string match, no LLM
judge).

    python scripts/run_behavioral_qa.py --model qwen3.5-0.8b-instruct --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from urdu_interp import models, paths, reporting


def _normalize(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[^\w\s]", "", text)


def _build_prompt(tokenizer, question: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return question


def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    generated_ids = out[0][enc["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _score(generated: str, aliases: list[str]) -> bool:
    norm_gen = _normalize(generated)
    return any(_normalize(alias) in norm_gen for alias in aliases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="an *instruct* model key from configs/models.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qa-file", default=str(paths.DATA_DIR / "qa" / "factual_qa.json"))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.qa_file, encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for item in items:
        for lang, question_key in (("ur", "question_ur"), ("hi", "question_hi")):
            prompt = _build_prompt(tokenizer, item[question_key])
            generated = _generate(model, tokenizer, prompt, args.device, max_new_tokens=args.max_new_tokens)
            correct = _score(generated, item["answer_aliases"])
            rows.append(
                {
                    "model": args.model,
                    "id": item["id"],
                    "topic": item["topic"],
                    "lang": lang,
                    "question": item[question_key],
                    "generated": generated,
                    "correct": correct,
                }
            )

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RESULTS_DIR / "behavioral_qa" / f"{args.model}.csv"
    reporting.save_table(df, out_path)

    for lang in ("ur", "hi"):
        sub = df[df["lang"] == lang]
        acc = sub["correct"].mean()
        print(f"{args.model} / {lang}: {int(sub['correct'].sum())}/{len(sub)} correct ({acc:.1%})")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

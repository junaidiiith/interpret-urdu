"""RQ2 follow-up EXP-6: does any of the internals-only "detour" story show up
in real generated text? Sidesteps every lens-calibration question in EXP-1
through EXP-5 by scoring actual model output: ask instruct models to answer
*only* in Urdu, generate a real paragraph (not RQ2-5's 5-token cap), and
measure code-switching two ways -- (a) word-level script classification
(script_utils.classify_token, the same deterministic Unicode-block classifier
RQ2 already uses, applied to the model's own words instead of a lens
reading), and (b) whole-text langid.classify as a coarser "did it just
default to a different language entirely" check.

    python scripts/run_exp6_behavioral_drift.py --model qwen3.5-0.8b-instruct --device cuda:2
"""
from __future__ import annotations

import argparse
import json

import langid
import torch

from urdu_interp import models, paths, reporting, script_utils

# Reuses factual_qa.json's questions (already vetted, disclosed non-expert
# construction, see MOT-6/results/qualitative_spotcheck.md) but asks for an
# elaborated multi-sentence answer instead of a one-word answer, so there's
# actually enough generated text to measure drift over. Adds a handful of
# open-ended prompts with no factual content at all, so the check isn't
# entirely QA-shaped.
_OPEN_PROMPTS = [
    "اپنی پسندیدہ موسم کے بارے میں تین جملے لکھیں۔",
    "دوستی کی اہمیت کے بارے میں مختصر لکھیں۔",
    "ایک اچھے دن کی تفصیل بیان کریں۔",
    "کتاب پڑھنے کے فائدے بیان کریں۔",
    "اپنے گھر کے بارے میں بتائیں۔",
    "صبح کے وقت کے بارے میں لکھیں۔",
]
_INSTRUCTION = "براہ کرم صرف اردو میں، تفصیل سے جواب دیں: "


def _build_prompt(tokenizer, question: str, enable_thinking: bool | None) -> str:
    # Some of these instruct models (confirmed: qwen3.5-*-instruct,
    # smollm3-3b-instruct) default to an extended reasoning ("<think>...")
    # scaffold whose *language* is a separate question from the final
    # answer's language -- conflating the two would misattribute "the
    # reasoning scaffold defaults to English" as "the final answer
    # code-switched." enable_thinking=False asks for a direct answer only,
    # which is the fairer comparison across models; enable_thinking=None
    # leaves the template's own default in place (used for a disclosed
    # secondary run, not the headline comparison).
    full_question = _INSTRUCTION + question
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": full_question}]
        kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return full_question


def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    generated_ids = out[0][enc["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _word_script_breakdown(text: str) -> dict:
    words = text.split()
    counts = {"urdu": 0, "hindi": 0, "latin": 0, "other": 0}
    for w in words:
        counts[script_utils.classify_token(w)] += 1
    scripted = counts["urdu"] + counts["hindi"] + counts["latin"]
    code_switch_rate = (counts["hindi"] + counts["latin"]) / scripted if scripted else float("nan")
    return {"n_words": len(words), **{f"n_{k}": v for k, v in counts.items()}, "code_switch_rate": code_switch_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="an *instruct* model key from configs/models.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qa-file", default=str(paths.DATA_DIR / "qa" / "factual_qa.json"))
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None, help="cap total prompts (QA + open) for a quick run")
    parser.add_argument(
        "--thinking",
        default="off",
        choices=["off", "default", "on"],
        help="'off' asks for a direct answer (enable_thinking=False where supported) -- the fair cross-model "
        "comparison; 'default' leaves the template's own default reasoning-mode setting in place; 'on' forces "
        "enable_thinking=True explicitly, for models (e.g. Qwen3.5) whose default already suppresses thinking "
        "-- lets the SmolLM3-vs-others thinking-mode comparison be made on equal footing",
    )
    args = parser.parse_args()
    enable_thinking = {"off": False, "default": None, "on": True}[args.thinking]

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.qa_file, encoding="utf-8") as f:
        qa_items = json.load(f)

    prompts = [(item["id"], "qa", item["question_ur"]) for item in qa_items] + [
        (f"open_{i}", "open", q) for i, q in enumerate(_OPEN_PROMPTS)
    ]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    rows = []
    for item_id, kind, question in prompts:
        prompt = _build_prompt(tokenizer, question, enable_thinking)
        generated = _generate(model, tokenizer, prompt, args.device, max_new_tokens=args.max_new_tokens)
        breakdown = _word_script_breakdown(generated)
        lang, score = langid.classify(generated) if generated.strip() else (None, None)
        rows.append(
            {
                "model": args.model,
                "id": item_id,
                "kind": kind,
                "question": question,
                "generated": generated,
                "langid_lang": lang,
                "langid_score": score,
                **breakdown,
            }
        )
        print(f"[{args.model}] {item_id} ({kind}): code_switch_rate={breakdown['code_switch_rate']:.2f}, langid={lang}")

    df = reporting.rows_to_dataframe(rows)
    suffix = "" if args.thinking == "off" else f"__thinking_{args.thinking}"
    out_path = paths.RESULTS_DIR / "exp6_behavioral_drift" / f"{args.model}{suffix}.csv"
    reporting.save_table(df, out_path)

    valid = df[df["code_switch_rate"].notna()]
    mean_rate = valid["code_switch_rate"].mean()
    frac_langid_non_ur = (df["langid_lang"] != "ur").mean()
    print(f"\n{args.model}: mean word-level code_switch_rate={mean_rate:.3f} over {len(valid)} generations")
    print(f"{args.model}: {frac_langid_non_ur:.1%} of generations' whole-text langid label was NOT 'ur'")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

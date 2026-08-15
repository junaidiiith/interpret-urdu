"""RQ3 step 1: LoRA-continued-pretrain a base model on Rekhta Urdu text, and
report held-out perplexity before/after as a sanity check that training
actually did something (this is not RQ3's real evidence -- see
run_rq3_eval.py for the paired convergence/detour comparison).

    python scripts/run_rq3_finetune.py --model qwen3.5-0.8b-base --limit 3000 --gpu 0

--gpu restricts the process to a single physical GPU via CUDA_VISIBLE_DEVICES
*before* torch is imported. This matters specifically for this script (unlike
run_rq1_convergence.py / run_rq2_logit_lens.py, which never hit this):
transformers' Trainer auto-wraps the model in torch.nn.DataParallel across
every CUDA device the process can see, ignoring a plain --device argument --
on a shared multi-GPU machine that silently turns a single-GPU fine-tuning
job into one that contends for memory on every other GPU currently in use,
and it will OOM once those other GPUs are already busy (confirmed: this
crashed on this machine's 4-GPU box before the --gpu restriction was added).
"""
from __future__ import annotations

import os
import sys


def _consume_gpu_arg(argv: list[str]) -> str | None:
    if "--gpu" not in argv:
        return None
    idx = argv.index("--gpu")
    value = argv[idx + 1]
    del argv[idx : idx + 2]
    return value


_gpu = _consume_gpu_arg(sys.argv[1:])
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import argparse  # noqa: E402

from urdu_interp import corpora, finetune, models, paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model key from configs/models.yaml")
    parser.add_argument("--limit", type=int, default=3000, help="number of Rekhta poems to train/eval on")
    parser.add_argument("--eval-holdout", type=int, default=300, help="poems held out from training, for perplexity")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=None, help="physical GPU index to restrict this process to (recommended on shared machines)")
    parser.add_argument("--output-dir-suffix", default="", help="appended to the checkpoint output dir name -- use for RQ3-4's hyperparameter-grid reruns so they never touch the canonical <model>-urdu-ft checkpoint")
    args = parser.parse_args()
    device = "cuda:0" if _gpu is not None else "cuda"

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    if spec.is_finetuned:
        raise SystemExit(f"{args.model} is already a fine-tuned entry; pass a base model key instead")

    triples = corpora.load_rekhta("ghazals", limit=args.limit + args.eval_holdout)
    train_triples, holdout_triples = triples[: args.limit], triples[args.limit :]
    train_texts = finetune.rekhta_training_texts(train_triples, lang="ur")
    holdout_texts = finetune.rekhta_training_texts(holdout_triples, lang="ur")

    model, tokenizer = models.load_model_and_tokenizer(spec, device=device)
    ppl_before = finetune.eval_perplexity(model, tokenizer, holdout_texts, block_size=args.block_size, device=device)
    print(f"held-out perplexity before fine-tuning: {ppl_before:.3f}")

    output_dir = paths.FINETUNED_MODELS_DIR / f"{args.model}-urdu-ft{args.output_dir_suffix}"
    manifest = finetune.run_lora_finetune(
        model,
        tokenizer,
        train_texts,
        str(output_dir),
        base_model_key=args.model,
        block_size=args.block_size,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    ppl_after = finetune.eval_perplexity(model, tokenizer, holdout_texts, block_size=args.block_size, device=device)
    print(f"held-out perplexity after fine-tuning: {ppl_after:.3f}")
    print(f"saved fine-tuned model + finetune_manifest.json to {manifest.output_dir}")


if __name__ == "__main__":
    main()

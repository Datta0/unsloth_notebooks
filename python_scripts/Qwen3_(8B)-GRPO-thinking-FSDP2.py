"""Multi-GPU (FSDP2) GRPO with Qwen3 NATIVE thinking enabled + colocate vLLM.

Companion to the ``enable_thinking=False`` recipes: this one keeps Qwen3's native
``<think>...</think>`` reasoning ON and rewards a thinking-native shape:
  * close ``</think>`` exactly once   (fast-learnable; implicitly rewards finishing
                                       the chain within the completion budget)
  * emit a ``\\boxed{answer}`` after it
  * numeric correctness of the boxed answer

Verified climb (Qwen3-8B, GSM8K, full-FT, 2 GPUs, 60 steps): think-format reward
1.1 -> ~2.0, correctness ~3.4 -> ~4.6/5.0, and the model learns to reason CONCISELY
(mean completion 760 -> ~370 tokens, clipped-ratio 0.35 -> 0.03), grad-norm bounded.

THE ONE THING THAT MATTERS for native thinking: the dataset's chains must FIT the
completion budget. GSM8K (grade-school) chains close ``</think>`` within ~1024 tokens,
so completions terminate, the reward has within-group variance, and it learns. On a
competition-hard set (e.g. DAPO-Math) the native chain exceeds any full-FT-affordable
budget, NOTHING terminates, every completion gets the same reward, ``reward_std`` -> 0,
and there is no gradient (it silently stalls). If you must train native-thinking on hard
data: use LoRA (``EXAMPLE_USE_LORA=1``) so a much larger ``EXAMPLE_MAX_COMPLETION`` fits.

Colocate vLLM via TRL (``use_vllm=True, vllm_mode="colocate"``): vLLM holds its own
weight copy; the trainer syncs the sharded FSDP2 weights in each step. Same packaged
helpers (``prepare_grpo_vllm`` / ``setup_grpo_vllm``) as the other GRPO scripts here.

Run on 2 GPUs:

    CUDA_VISIBLE_DEVICES=4,5 \
      torchrun --standalone --nproc_per_node=2 "Qwen3_(8B)-GRPO-thinking-FSDP2.py"

Env knobs (all optional):
    EXAMPLE_MODEL=unsloth/Qwen3-8B        EXAMPLE_DATASET=gsm8k   (or: dapo)
    EXAMPLE_USE_LORA=0                    EXAMPLE_LORA_RANK=32
    EXAMPLE_MAX_STEPS=60                  EXAMPLE_MAX_COMPLETION=1024
    EXAMPLE_MAX_SEQ_LEN=2048              EXAMPLE_VLLM_GPU_MEM_UTIL=0.35
    EXAMPLE_REPORT_TO=wandb
"""
from __future__ import annotations

import os
import re

import unsloth  # noqa: F401  (import before transformers/trl)
import torch

from unsloth_dist import (
    FSDP2Config,
    get_rank,
    is_main_process,
    prepare_fsdp2,
    save_pretrained_fsdp2,
)
from unsloth_dist.grpo_colocate import prepare_grpo_vllm, setup_grpo_vllm

MODEL = os.environ.get("EXAMPLE_MODEL", "unsloth/Qwen3-8B")
DATASET = os.environ.get("EXAMPLE_DATASET", "gsm8k")  # gsm8k (chains fit budget) | dapo (hard)
MAX_SEQ_LEN = int(os.environ.get("EXAMPLE_MAX_SEQ_LEN", "2048"))
MAX_COMPLETION = int(os.environ.get("EXAMPLE_MAX_COMPLETION", "1024"))
MAX_STEPS = int(os.environ.get("EXAMPLE_MAX_STEPS", "60"))
MAX_TRAIN = int(os.environ.get("EXAMPLE_MAX_TRAIN", "2048"))
USE_LORA = os.environ.get("EXAMPLE_USE_LORA", "0") == "1"
LORA_RANK = int(os.environ.get("EXAMPLE_LORA_RANK", "32"))
VLLM_GPU_MEM_UTIL = float(os.environ.get("EXAMPLE_VLLM_GPU_MEM_UTIL", "0.35"))
OUTPUT_DIR = os.environ.get(
    "EXAMPLE_OUTPUT_DIR",
    "/mnt/disks/unslothai/datta0/cache/fsdp2_clean_pack/nb_grpo_thinking",
)

SYSTEM_PROMPT = (
    "Solve the math problem. Reason through it, then give the final answer "
    "on its own line as \\boxed{<answer>}."
)

_boxed = re.compile(r"\\boxed\{([^{}]*)\}")
_THINK_CLOSE = "</think>"


def _last_boxed(text: str):
    m = _boxed.findall(text)
    return m[-1].strip() if m else None


# --------------------------------------------------------------------------- #
# Rewards (thinking-native): fast-learnable format term + slow correctness term.
# --------------------------------------------------------------------------- #
def think_format_reward(completions, **kwargs):
    """+1 for closing </think> exactly once (else -1: open == clipped mid-think),
    +1 for a \\boxed{} after the close (else -0.5). Range ~ -1.5 .. +2.0."""
    scores = []
    for c in completions:
        resp = c[0]["content"]
        score = 1.0 if resp.count(_THINK_CLOSE) == 1 else -1.0
        tail = resp.split(_THINK_CLOSE, 1)[1] if _THINK_CLOSE in resp else resp
        score += 1.0 if _boxed.search(tail) is not None else -0.5
        scores.append(score)
    return scores


def correctness_reward(prompts, completions, answer, **kwargs):
    """Numeric correctness of the boxed answer: +5 exact, ratio partial, -2 if none."""
    extracted = [_last_boxed(c[0]["content"]) for c in completions]
    scores = []
    for guess, gold in zip(extracted, answer):
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == gold or guess.strip() == str(gold).strip():
            scores.append(5.0)
            continue
        try:
            ratio = float(guess.replace(",", "")) / float(str(gold).replace(",", ""))
            scores.append(2.0 if 0.9 <= ratio <= 1.1 else 1.5 if 0.8 <= ratio <= 1.2 else -2.5)
        except Exception:
            scores.append(-2.5)
    return scores


def build_dataset():
    from datasets import load_dataset

    if DATASET == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        mapper = lambda x: {  # noqa: E731
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": x["question"]}],
            "answer": x["answer"].split("####")[-1].strip().replace(",", ""),
        }
    else:  # dapo (hard — see the module docstring; needs a large budget / LoRA)
        ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
        mapper = lambda x: {  # noqa: E731
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": x["prompt"]}],
            "answer": x["solution"],
        }
    if MAX_TRAIN > 0:
        ds = ds.select(range(min(MAX_TRAIN, len(ds))))
    return ds.map(mapper, remove_columns=[c for c in ds.column_names if c not in ("prompt", "answer")])


def main():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=False,
        full_finetuning=not USE_LORA, device_map={"": local_rank},
    )
    if USE_LORA:
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    train_dataset = build_dataset()

    model = prepare_fsdp2(model, FSDP2Config(compile="off"))
    prepare_grpo_vllm(rank=rank, is_lora=USE_LORA, lora_rank=LORA_RANK)

    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        vllm_importance_sampling_correction=False,
        temperature=1.0,
        top_k=-1,
        beta=0.0,
        scale_rewards="group",
        num_iterations=1,
        num_generations=4,
        max_completion_length=MAX_COMPLETION,
        # Qwen3 native thinking ON: the template leaves the assistant turn open so the
        # model generates its own <think>...</think> (do NOT set enable_thinking=False,
        # which pre-injects an empty, pre-closed think block -> the model never reasons).
        chat_template_kwargs={"enable_thinking": True},
        per_device_train_batch_size=4,   # divisible by num_generations
        gradient_accumulation_steps=2,
        learning_rate=2e-4 if USE_LORA else 5e-6,
        lr_scheduler_type="constant_with_warmup" if USE_LORA else "cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.1,
        weight_decay=0.001,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_steps=MAX_STEPS,
        seed=3407,
        data_seed=3407,
        bf16=not USE_LORA,
        gradient_checkpointing=True,
        optim="adamw_torch",  # full-FT: auto-swapped to torchao SR-AdamW by setup_grpo_vllm
        logging_steps=1,
        report_to=os.environ.get("EXAMPLE_REPORT_TO", "none"),
        run_name=os.environ.get("WANDB_NAME", "grpo_thinking_qwen3_8b"),
        save_strategy="no",
        disable_tqdm=not is_main_process(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[think_format_reward, correctness_reward],
        args=args,
        train_dataset=train_dataset,
    )
    setup_grpo_vllm(trainer, rank=rank, is_lora=USE_LORA)

    trainer.train()

    save_pretrained_fsdp2(model, OUTPUT_DIR, tokenizer=tok,
                          save_method="lora" if USE_LORA else "full")
    if is_main_process():
        print(f"[grpo_thinking] done; checkpoint at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

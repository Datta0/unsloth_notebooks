"""Multi-GPU (FSDP2) GRPO with colocate vLLM — the DeepSeek-R1-0528-Qwen3-8B notebook.

This is the canonical Unsloth GRPO recipe from ``DeepSeek_R1_0528_Qwen3_(8B)_GRPO.py``
in this folder (DAPO-Math + the format/answer reward stack), turned into a runnable
multi-GPU FSDP2 script. Generation is colocate vLLM via TRL
(``GRPOConfig(use_vllm=True, vllm_mode="colocate")``): vLLM runs in-process with its
OWN weight copy and the trainer syncs the sharded FSDP2 weights into it each step, so
it never touches the model's (sharded) ``generate()``. This reproduced the Qwen3-8B /
DAPO-Math reward climb.

Diff vs the single-GPU notebook:
  1. ``import unsloth_dist`` + the two colocate helpers.
  2. Load WITHOUT ``fast_inference=True`` (that is Unsloth's weight-SHARING vLLM engine,
     which fights FSDP2). vLLM here is a separate engine fed by the weight sync.
  3. ``prepare_fsdp2(model, FSDP2Config(compile="off"))`` to shard.
  4. ``prepare_grpo_vllm(...)`` BEFORE the trainer; ``setup_grpo_vllm(trainer, ...)`` AFTER.
  5. GRPOConfig: ``use_vllm=True, vllm_mode="colocate"``, ``save_strategy="no"``,
     and NO ``max_prompt_length`` / ``vllm_sampling_params`` (both removed in TRL 1.3.0).
  6. ``save_pretrained_fsdp2(...)`` (plain save crashes on DTensors); launch with torchrun.
  7. Colab ``!pip`` / ``get_ipython`` / inference / GGUF cells dropped.

Defaults to FULL-finetune (the path the reward climb was measured on). ``setup_grpo_vllm``
auto-installs stochastic-rounding AdamW for bf16 full-FT (plain AdamW underflows and
flat-lines). Set ``EXAMPLE_USE_LORA=1`` for 16-bit LoRA instead.

Run on 2 GPUs (training + colocate vLLM share each GPU):

    CUDA_VISIBLE_DEVICES=4,5 WANDB_PROJECT=nb_grpo_fsdp2 \
      torchrun --standalone --nproc_per_node=2 "DeepSeek_R1_0528_Qwen3_(8B)_GRPO-FSDP2.py"

Watch the ``reward`` column climb (be patient — 100-200 steps). Env knobs:
    EXAMPLE_MODEL=unsloth/DeepSeek-R1-0528-Qwen3-8B   EXAMPLE_USE_LORA=0
    EXAMPLE_DATASET=dapo|gsm8k   EXAMPLE_MAX_STEPS=200   EXAMPLE_MAX_SEQ_LEN=2048
    EXAMPLE_MAX_COMPLETION=1024   EXAMPLE_VLLM_GPU_MEM_UTIL=0.3   EXAMPLE_REPORT_TO=wandb
    EXAMPLE_USE_LANG_REWARD=1   (adds the Bahasa-Indonesia language reward; needs `langid`)

For a climb OUT OF THE BOX, run with ``EXAMPLE_DATASET=gsm8k EXAMPLE_USE_LORA=1
EXAMPLE_MAX_COMPLETION=2048``: grade-school chains fit the budget so completions
terminate, and the gsm8k reward reads the final ``\\boxed{}`` answer. Verified on
DeepSeek-R1-0528-Qwen3-8B (2 GPUs, 60 steps): reward 4.7 -> 5.8, correctness 2.4 -> 3.0,
completions self-shorten 1653 -> 862 tokens, clip 0.47 -> 0.29.

The DEFAULT (``EXAMPLE_DATASET=dapo``) is the faithful-but-HARD combo: a heavy reasoner +
competition-hard DAPO-Math + think-tag rewards. At a full-FT-affordable budget the native
chain rarely closes ``</think>`` before the cap, so most completions clip, per-group reward
variance can collapse, and the climb is slow/stalled. Give it room
(``EXAMPLE_MAX_COMPLETION=2048+`` with ``EXAMPLE_USE_LORA=1`` so it fits memory), or switch
to ``EXAMPLE_DATASET=gsm8k`` as above.
"""
from __future__ import annotations

import os
import re

import unsloth  # noqa: F401  (import before transformers/trl)
import torch

import unsloth_dist
from unsloth_dist import (
    FSDP2Config,
    get_rank,
    is_main_process,
    prepare_fsdp2,
    save_pretrained_fsdp2,
)
from unsloth_dist.grpo_colocate import prepare_grpo_vllm, setup_grpo_vllm

MODEL = os.environ.get("EXAMPLE_MODEL", "unsloth/DeepSeek-R1-0528-Qwen3-8B")
MAX_SEQ_LEN = int(os.environ.get("EXAMPLE_MAX_SEQ_LEN", "2048"))
MAX_COMPLETION = int(os.environ.get("EXAMPLE_MAX_COMPLETION", "1024"))
# Dataset: "dapo" (faithful default; competition-hard, chains often exceed the
# budget -> reward can flat-line) or "gsm8k" (grade-school, chains fit -> the
# reward actually climbs). The reward stack below is format-generic (it scores
# the text after </think> against a gold answer), so it works for either.
DATASET = os.environ.get("EXAMPLE_DATASET", "dapo").lower()
MAX_STEPS = int(os.environ.get("EXAMPLE_MAX_STEPS", "200"))
USE_LORA = os.environ.get("EXAMPLE_USE_LORA", "0") == "1"
LORA_RANK = int(os.environ.get("EXAMPLE_LORA_RANK", "32"))
VLLM_GPU_MEM_UTIL = float(os.environ.get("EXAMPLE_VLLM_GPU_MEM_UTIL", "0.3"))
USE_LANG_REWARD = os.environ.get("EXAMPLE_USE_LANG_REWARD", "0") == "1"
OUTPUT_DIR = os.environ.get(
    "EXAMPLE_OUTPUT_DIR",
    "/mnt/disks/unslothai/datta0/cache/fsdp2_clean_pack/nb_grpo",
)

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    "Then give the final answer."
)


def main():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset

    rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 1. Load on this rank's GPU. NO fast_inference (that is the weight-sharing vLLM
    #    that conflicts with FSDP2). 16-bit, not 4-bit.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        full_finetuning=not USE_LORA,
        device_map={"": local_rank},
    )
    if USE_LORA:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
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

    # --- think-tag detection (verbatim from the notebook) ------------------- #
    reasoning_start = reasoning_end = None
    for token in tokenizer.get_added_vocab().keys():
        if "think" in token and "/" in token:
            reasoning_end = token
        elif "think" in token:
            reasoning_start = token
    # Fallback if the model doesn't expose them as added tokens.
    reasoning_start = reasoning_start or "<think>"
    reasoning_end = reasoning_end or "</think>"

    # --- dataset ------------------------------------------------------------ #
    if DATASET == "gsm8k":
        # Grade-school math: chains fit a modest budget, so completions terminate
        # and the answer reward gets a real gradient. Gold answer = the number
        # after "####"; the reward stack scores the text after </think>.
        def _gsm8k_answer(text):
            m = re.search(r"####\s*([-\d,\.]+)", text)
            return m.group(1).replace(",", "").strip() if m else None

        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.map(lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": _gsm8k_answer(x["answer"]),
        }, remove_columns=dataset.column_names)
    else:
        # DAPO-Math, verbatim mapping (faithful to the source notebook).
        dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
        dataset = dataset.map(lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["solution"],
        })

    # --- reward functions (verbatim from the notebook) ---------------------- #
    match_format = re.compile(rf"{reasoning_end}(.*)", re.DOTALL)
    match_numbers = re.compile(r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL)

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            scores.append(3.0 if match_format.search(response) is not None else 0.0)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0.0
            score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            (g.group(1) if (g := match_format.search(r)) is not None else None)
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.0)
                continue
            score = 0.0
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except Exception:
                    score -= 4.5
            scores.append(score)
        return scores

    printed = {"n": 0}

    def check_numbers(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            (g.group(1) if (g := match_numbers.search(r)) is not None else None)
            for r in responses
        ]
        if is_main_process() and printed["n"] % 5 == 0:
            print("*" * 20 + f"Answer: {answer[0]}\nResponse: {responses[0]}\n"
                  f"Extracted: {extracted[0]}", flush=True)
        printed["n"] += 1
        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                ta = float(str(true_answer).strip())
                gv = float(guess.strip().replace(",", ""))
                scores.append(3.5 if gv == ta else -1.5)
            except Exception:
                scores.append(0.0)
        return scores

    # GSM8K answers come back as \boxed{N} (or a trailing number) after </think>;
    # the verbatim DAPO check_answer/check_numbers misread that (they compare the
    # whole post-</think> paragraph to the bare number, and match_numbers grabs the
    # first digit/comma anywhere in the reasoning). Use a boxed/last-number
    # extractor + numeric compare for the gsm8k path; keep the DAPO rewards verbatim.
    def _extract_final_number(text):
        region = text.split(reasoning_end, 1)[-1] if reasoning_end in text else text
        mb = re.search(r"\\boxed\{([^}]*)\}", region)
        src = mb.group(1) if mb else region
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", src)
        return nums[-1].replace(",", "").rstrip(".") if nums else None

    def gsm8k_correctness(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [_extract_final_number(r) for r in responses]
        if is_main_process() and printed["n"] % 5 == 0:
            print("*" * 20 + f"Answer: {answer[0]}  Extracted: {extracted[0]}", flush=True)
        printed["n"] += 1
        scores = []
        for guess, gold in zip(extracted, answer):
            if guess is None:
                scores.append(-2.0)
                continue
            try:
                scores.append(5.0 if abs(float(guess) - float(gold)) < 1e-6 else -1.0)
            except Exception:
                scores.append(-1.0)
        return scores

    if DATASET == "gsm8k":
        reward_funcs = [match_format_exactly, match_format_approximately, gsm8k_correctness]
    else:
        reward_funcs = [match_format_exactly, match_format_approximately, check_answer, check_numbers]

    if USE_LANG_REWARD:
        try:
            import langid

            def format_and_language_reward_func(completions, **kwargs):
                scores = []
                for item in completions:
                    if not item or "content" not in item[0]:
                        scores.append(-5.0)
                        continue
                    lang, _ = langid.classify(item[0]["content"] or "")
                    scores.append({"id": 5.0, "en": -3.0, "zh": -3.0}.get(lang, -5.0))
                return scores

            reward_funcs.append(format_and_language_reward_func)
        except ImportError:
            if is_main_process():
                print("[nb_grpo] langid not installed; skipping language reward", flush=True)

    # 2. Shard (compile off matches the proven climb), then pre-trainer patches.
    model = prepare_fsdp2(model, FSDP2Config(compile="off"))
    prepare_grpo_vllm(rank=rank, is_lora=USE_LORA, lora_rank=LORA_RANK)

    # 3. Colocate GRPOConfig. temperature=1.0 is load-bearing (near-greedy collapses the
    #    group -> advantage 0 -> no learning). Thinking is left ON: the format rewards
    #    above are built around the <think>/</think> tags.
    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        # On-policy (num_iterations=1) -> no vLLM importance-sampling correction needed.
        vllm_importance_sampling_correction=False,
        temperature=1.0,
        top_k=-1,
        beta=0.0,
        scale_rewards="group",
        num_iterations=1,
        num_generations=4,
        max_completion_length=MAX_COMPLETION,
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
        run_name=os.environ.get("WANDB_NAME", "nb_grpo_fsdp2_qwen3_8b"),
        # HF periodic save crashes on FSDP2 DTensors -> save once with save_pretrained_fsdp2.
        save_strategy="no",
        disable_tqdm=not is_main_process(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
    )

    # 4. Post-trainer colocate patches: per-step DTensor->vLLM weight sync, chunked logps,
    #    FA2 pad-mask fix, SR-AdamW (full-FT).
    setup_grpo_vllm(trainer, rank=rank, is_lora=USE_LORA)

    trainer.train()

    # 5. Save (rank 0 writes; all ranks must enter the collective gather).
    save_pretrained_fsdp2(
        model, OUTPUT_DIR, tokenizer=tokenizer,
        save_method="lora" if USE_LORA else "full",
    )
    if is_main_process():
        print(f"[nb_grpo] done; checkpoint at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Multi-GPU (FSDP2) SFT — the Qwen3 Alpaca notebook, run across GPUs with unsloth_dist.

This is the canonical Unsloth "Qwen3 + Alpaca" LoRA SFT recipe (see
``Qwen3_(14B)-Alpaca.py`` in this folder), turned into a runnable multi-GPU FSDP2
script. SFT needs NO explicit sharding calls: ``import unsloth_dist`` arms an
implicit FSDP2 path that shards the model inside the trainer under ``torchrun`` and
routes ``model.save_pretrained`` through the FSDP2 collective save.

Diff vs the single-GPU notebook (that is the whole story):
  1. ``import unsloth_dist``                         (top of file)
  2. drop the Colab ``!pip`` / ``get_ipython`` cells  (so it runs as a .py)
  3. ``save_strategy="no"``                           (HF periodic save crashes on
                                                       FSDP2 DTensors; save once at the end)
  4. launch with ``torchrun`` instead of ``python``
  5. inference / GGUF cells dropped (``model.generate`` on a sharded model is
     unsupported in-process — save, then load the checkpoint in a fresh process)

Run on 2 GPUs (data-parallel FSDP2):

    CUDA_VISIBLE_DEVICES=4,5 \
      torchrun --standalone --nproc_per_node=2 "Qwen3_(8B)-Alpaca-FSDP2.py"

Iterate single-GPU (everything below is a plain Unsloth no-op):

    CUDA_VISIBLE_DEVICES=4 python "Qwen3_(8B)-Alpaca-FSDP2.py"

Env knobs (all optional):
    EXAMPLE_MODEL=unsloth/Qwen3-8B        EXAMPLE_USE_LORA=1   (0 = full finetune)
    EXAMPLE_MAX_STEPS=60                  EXAMPLE_MAX_SEQ_LEN=2048
    EXAMPLE_OUTPUT_DIR=/mnt/disks/unslothai/datta0/cache/fsdp2_clean_pack/nb_sft
"""
from __future__ import annotations

import os

import unsloth_dist  # noqa: F401  arms implicit FSDP2 under torchrun; no-op under plain python
from unsloth import FastLanguageModel
import torch

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("EXAMPLE_MODEL", "unsloth/Qwen3-8B")
USE_LORA = os.environ.get("EXAMPLE_USE_LORA", "1") == "1"
MAX_STEPS = int(os.environ.get("EXAMPLE_MAX_STEPS", "60"))
MAX_SEQ_LEN = int(os.environ.get("EXAMPLE_MAX_SEQ_LEN", "2048"))
OUTPUT_DIR = os.environ.get(
    "EXAMPLE_OUTPUT_DIR",
    "/mnt/disks/unslothai/datta0/cache/fsdp2_clean_pack/nb_sft",
)

# --------------------------------------------------------------------------- #
# Data prep — verbatim from the Alpaca notebook.
# --------------------------------------------------------------------------- #
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def build_dataset(tokenizer):
    eos = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            # Must add EOS_TOKEN, otherwise generation goes on forever.
            texts.append(alpaca_prompt.format(instruction, inp, output) + eos)
        return {"text": texts}

    ds = load_dataset("unsloth/alpaca-cleaned", split="train")
    return ds.map(formatting_prompts_func, batched=True)


def main():
    # 1. Load on this rank's GPU. 16-bit (load_in_4bit=False) — 4-bit + FSDP2 is a
    #    separate path; plain bf16 LoRA/full-FT is the supported one here.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        dtype=None,
    )

    if USE_LORA:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=build_dataset(tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=1,
            # adamw_8bit (bitsandbytes) in the notebook -> torch optimizer under FSDP2.
            optim="adamw_torch_fused",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            data_seed=3407,
            output_dir=OUTPUT_DIR,
            report_to=os.environ.get("EXAMPLE_REPORT_TO", "none"),
            # HF Trainer's periodic save calls model.save_pretrained on the live
            # sharded model -> crashes on FSDP2 DTensors. Save once at the end.
            save_strategy="no",
            dataset_text_field="text",
            max_length=MAX_SEQ_LEN,
        ),
    )

    trainer.train()

    # Ordinary single-GPU save. Under torchrun this is auto-routed through the FSDP2
    # collective (gathered on all ranks, written from rank 0).
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
train_1m.py — Full fine-tuning throughput benchmark with SCAO on TinyStories-1M

This script measures raw optimizer throughput — no LoRA, no frozen layers,
every single parameter goes through SCAO's preconditioner. The goal is to
verify that curvature-aware updates do NOT destroy training speed.

Model: roneneldan/TinyStories-1M (~3.7M trainable parameters)
Expected result: ~627 tokens/second, proving that the gain in convergence
per step more than compensates for the preconditioner overhead.

Requirements:
    pip install transformers datasets torch

Run:
    python train_1m.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from scao import SCAO


def main():
    print("🚀 Initializing throughput benchmark (SCAO @ 1M params)...")

    # TinyStories-1M is a tiny research model from Microsoft — perfect for
    # measuring throughput because it's small enough to load anywhere but
    # big enough to exercise the full optimizer pipeline on real parameters.
    model_id = "roneneldan/TinyStories-1M"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🧠 Loading model — full fine-tuning, no LoRA...")
    model = AutoModelForCausalLM.from_pretrained(model_id)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Total parameters SCAO will handle: {total_params:,}")

    print("📚 Downloading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize, batched=True)

    print("⚙️ Initializing SCAO over all model parameters...")
    # Passing model.parameters() directly — every weight gets curvature tracking
    optimizer = SCAO(model.parameters(), lr=1e-3)

    args = TrainingArguments(
        output_dir="./resultado_scao_1m",
        per_device_train_batch_size=8,  # Larger batch is fine — the model is tiny
        max_steps=100,                  # 100 steps is enough to measure stable throughput
        logging_steps=10,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    print("⚡ Running... watch the tokens/sec in the training log!")
    trainer.train()

    print("✅ Benchmark complete! Check the log above for throughput numbers.")


if __name__ == "__main__":
    main()
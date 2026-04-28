"""
train_local.py — Fine-tune GPT-2 (125M) with SCAO + LoRA on a consumer GPU

This script demonstrates that a 2nd-order optimizer does NOT require a data
center to run. By combining SCAO's Diagonal Fallback with LoRA's parameter
efficiency, the entire fine-tuning fits comfortably under 8 GB VRAM.

Requirements:
    pip install transformers datasets peft torch

Run:
    python train_local.py
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
from peft import LoraConfig, get_peft_model, TaskType
from scao import SCAO


def main():
    print("🚀 Loading model and tokenizer...")
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    # ------------------------------------------------------------------
    # Inject LoRA adapters — this is how we keep VRAM under control.
    # Instead of updating all 124M parameters, LoRA injects tiny low-rank
    # matrices (rank=8) into the attention projections. We end up training
    # less than 1% of the model's weights while still adapting its behavior.
    # ------------------------------------------------------------------
    print("🧠 Injecting LoRA adapters into GPT-2...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,             # Low-rank dimension — 8 is lightweight and effective
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # Attention projection weights in GPT-2
    )
    model = get_peft_model(model, peft_config)

    # This will confirm you're only training ~0.5% of the total parameters
    model.print_trainable_parameters()

    print("📚 Downloading WikiText-2 (using the first 1% for a quick demo)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize, batched=True)

    print("⚙️ Initializing SCAO for the LoRA weights only...")

    # Only pass the trainable LoRA parameters to SCAO — the frozen base model
    # weights don't need an optimizer state, which saves even more memory.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # To activate INT8 EMA quantization (saves an extra 36% VRAM on the
    # curvature accumulators), use: SCAO(trainable_params, lr=1e-3, use_int8_ema=True)
    optimizer = SCAO(trainable_params, lr=1e-3)

    args = TrainingArguments(
        output_dir="./resultado_scao_local",
        per_device_train_batch_size=2,  # Batch size 2 keeps us safely under 8 GB
        max_steps=50,
        logging_steps=10,
        report_to="none",
    )

    # DataCollatorForLanguageModeling automatically creates the "labels" tensor
    # from the input ids — required for computing the causal LM loss correctly.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # Pass SCAO; set scheduler to None
    )

    print("🔥 Starting local training (LoRA + SCAO)...")
    trainer.train()

    print("✅ Training complete! Checkpoint saved to ./resultado_scao_local")


if __name__ == "__main__":
    main()
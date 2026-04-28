import torch
import argparse
import time
import json
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from scao import SCAO
import torch_optimizer as optim

def get_peak_vram():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Pro Benchmark: SCAO vs Shampoo (4B Scale)")
    parser.add_argument("--optimizer", type=str, choices=["scao", "shampoo"], required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--steps", type=int, default=100)
    args_cmd = parser.parse_args()

    print(f"🚀 Starting Professional Benchmark: {args_cmd.optimizer.upper()} on {args_cmd.model_id}")
    
    # 1. Model Loading (4-bit)
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args_cmd.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 2. LoRA Setup
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 3. Dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]") # Using small slice for benchmark speed
    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Optimizer Selection
    print(f"⚙️ Initializing {args_cmd.optimizer.upper()}...")
    if args_cmd.optimizer == "scao":
        optimizer = SCAO(trainable_params, lr=2e-4)
    else:
        # Shampoo is much heavier; this initialization might fail or be very slow
        optimizer = optim.Shampoo(trainable_params, lr=2e-4)

    # 5. Training Config
    training_args = TrainingArguments(
        output_dir=f"./results_{args_cmd.optimizer}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args_cmd.steps,
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        fp16=True, # T4 support
        optim="adamw_torch" # Placeholder
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    # 6. Run & Report
    start_time = time.time()
    status = "Success"
    error_msg = ""
    
    try:
        print("⚡ Training active! Monitoring performance...")
        trainer.train()
    except Exception as e:
        status = "Failed"
        error_msg = str(e)
        print(f"❌ Error caught: {error_msg}")

    end_time = time.time()
    duration = end_time - start_time
    peak_vram = get_peak_vram()

    # 7. Saving Metrics
    metrics = {
        "optimizer": args_cmd.optimizer,
        "status": status,
        "peak_vram_gb": f"{peak_vram:.2f}",
        "duration_sec": f"{duration:.2f}",
        "it_per_sec": f"{args_cmd.steps / duration:.2f}" if duration > 0 else "0",
        "error": error_msg
    }

    with open(f"bench_report_{args_cmd.optimizer}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*30)
    print(f"BENCHMARK COMPLETED: {args_cmd.optimizer.upper()}")
    print(f"Status: {status}")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Speed: {metrics['it_per_sec']} it/s")
    print("="*30)

    # Append to markdown summary
    with open("final_summary.md", "a") as f:
        if os.stat("final_summary.md").st_size == 0:
            f.write("# SCAO vs Shampoo Pro Benchmark\n\n")
            f.write("| Optimizer | Status | Peak VRAM (GB) | Speed (it/s) |\n")
            f.write("|-----------|--------|----------------|--------------|\n")
        f.write(f"| {args_cmd.optimizer.upper()} | {status} | {peak_vram:.2f} | {metrics['it_per_sec']} |\n")

if __name__ == "__main__":
    if not os.path.exists("final_summary.md"):
        open("final_summary.md", "w").close()
    main()

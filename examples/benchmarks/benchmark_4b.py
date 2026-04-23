import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from scao import SCAO # Your 2nd-order optimizer implementation

def main():
    print("🚀 Starting 4B-Scale Benchmark for SCAO...")
    
    # Using Qwen 2.5 3B model (optimal for the 4B category tests)
    model_id = "Qwen/Qwen2.5-3B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📦 Loading base model in 4-bit (QLoRA) to optimize GPU memory usage...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("🧠 Initializing LoRA adapters...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Focusing on Attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Filter only parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"🔥 Trainable parameters (LoRA): {sum(p.numel() for p in trainable_params):,}")

    print("📚 Loading dataset...")
    # Using wikitext-2 for consistency across benchmarks
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    
    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)

    print("⚙️ Injecting SCAO Optimizer...")
    # SCAO uses 2nd-order information for faster convergence
    optimizer = SCAO(trainable_params, lr=2e-4) # Standard QLoRA learning rate

    # Data collator for causal language modeling (automatically creates 'labels')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="./scao_benchmark_4b_results",
        per_device_train_batch_size=2, # Small batch size to manage VRAM constraints
        gradient_accumulation_steps=4, # Effective batch size of 8
        max_steps=100,                 # 100 steps to evaluate performance and loss decay
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,   # Essential to avoid Out-Of-Memory errors
        optim="adamw_torch"            # Placeholder; SCAO will override this if passed to Trainer
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,   # Pass the collator to handle sequence labeling
        # To fully utilize SCAO, pass it to the Trainer's optimizers argument:
        optimizers=(optimizer, None)
    )

    print("⚡ Training active! Watch for the loss reduction curve...")
    trainer.train()

if __name__ == "__main__":
    main()

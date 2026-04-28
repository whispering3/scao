import argparse
import time
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch_optimizer as optim
from scao import SCAO

class BenchmarkLogger:
    def __init__(self, optimizer_name, test_type):
        self.optimizer_name = optimizer_name
        self.test_type = test_type
        self.results = {
            "optimizer": optimizer_name,
            "test_type": test_type,
            "status": "Incomplete",
            "metrics": {},
            "errors": None,
            "logs": []
        }

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        print(formatted_msg)
        self.results["logs"].append(formatted_msg)

    def save_report(self):
        filename = f"report_{self.optimizer_name}_{self.test_type}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)
        
        # Generate Markdown Summary
        md_filename = "benchmark_summary.md"
        exists = os.path.exists(md_filename)
        with open(md_filename, "a" if exists else "w") as f:
            if not exists:
                f.write("# SCAO vs Shampoo Benchmark Summary\n\n")
                f.write("| Optimizer | Test | Status | Final Loss | Throughput (it/s) | Peak VRAM (GB) |\n")
                f.write("|-----------|------|--------|------------|-------------------|----------------|\n")
            
            m = self.results["metrics"]
            f.write(f"| {self.optimizer_name.upper()} | {self.test_type.upper()} | {self.results['status']} | {m.get('final_loss', 'N/A')} | {m.get('throughput', 'N/A')} | {m.get('peak_vram', 'N/A')} |\n")

def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0

def prepare_model(model_id, logger):
    logger.log(f"Loading model: {model_id} (4-bit QLoRA)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    return model, tokenizer

def run_stress_test(optimizer_type):
    logger = BenchmarkLogger(optimizer_type, "stress")
    logger.log("Starting Stress Test: Death Benchmark (3B Model)")
    
    try:
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        model, tokenizer = prepare_model(model_id, logger)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.log(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
        
        if optimizer_type == "shampoo":
            optimizer = optim.Shampoo(trainable_params, lr=1e-4)
        else:
            optimizer = SCAO(trainable_params, lr=1e-4)

        logger.log("Running forward/backward pass...")
        inputs = tokenizer("Benchmarking memory limits for high-order optimization.", return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        
        logger.log("Executing optimizer.step()...")
        optimizer.step()
        logger.results["status"] = "Success"
        
    except RuntimeError as e:
        logger.results["status"] = "Failed (OOM/Instability)"
        logger.results["errors"] = str(e)
        logger.log(f"Caught expected error: {str(e)[:100]}...")
    except Exception as e:
        logger.results["status"] = "Error"
        logger.results["errors"] = str(e)
        logger.log(f"Unexpected error: {e}")
    finally:
        logger.results["metrics"]["peak_vram"] = f"{get_peak_memory():.2f}"
        logger.save_report()

def run_convergence_test(optimizer_type, steps=200):
    logger = BenchmarkLogger(optimizer_type, "convergence")
    logger.log(f"Starting Convergence Test: 0.5B Model ({steps} steps)")
    
    try:
        model_id = "Qwen/Qwen2.5-0.5B"
        model, tokenizer = prepare_model(model_id, logger)
        
        logger.log("Loading dataset: wikitext...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokenized_datasets = dataset.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128),
            batched=True,
            remove_columns=["text"]
        ).filter(lambda x: len(x["input_ids"]) > 0)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if optimizer_type == "shampoo":
            optimizer = optim.Shampoo(trainable_params, lr=1e-4)
        else:
            optimizer = SCAO(trainable_params, lr=1e-4)

        model.train()
        model.gradient_checkpointing_enable()
        
        from torch.utils.data import DataLoader
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(tokenized_datasets, batch_size=1)
        
        start_time = time.time()
        last_loss = 0
        logger.log("Training loop started. The first step might take longer due to optimizer initialization.")
        
        for i, batch in enumerate(dataloader):
            if i >= steps: break
            
            if i < 5 or i % 20 == 0:
                logger.log(f"Step {i} - Forward/Backward...")
                
            inputs = batch['input_ids'].to(model.device)
            mask = batch['attention_mask'].to(model.device)
            
            outputs = model(input_ids=inputs, attention_mask=mask, labels=inputs)
            loss = outputs.loss
            loss.backward()
            
            if i < 5 or i % 20 == 0:
                logger.log(f"Step {i} - Optimizer step...")
                
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            last_loss = loss.item()
            if i % 20 == 0:
                logger.log(f"Step {i}/{steps} - Loss: {last_loss:.4f} - Peak VRAM: {get_peak_memory():.2f} GB")

        end_time = time.time()
        duration = end_time - start_time
        
        logger.results["status"] = "Success"
        logger.results["metrics"] = {
            "final_loss": f"{last_loss:.4f}",
            "throughput": f"{steps/duration:.2f}",
            "peak_vram": f"{get_peak_memory():.2f}"
        }
        
    except Exception as e:
        logger.results["status"] = "Failed"
        logger.results["errors"] = str(e)
        logger.log(f"Error during training: {e}")
    finally:
        logger.save_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional SCAO vs Shampoo Benchmark")
    parser.add_argument("--test", type=str, choices=["stress", "convergence"], required=True)
    parser.add_argument("--optimizer", type=str, choices=["shampoo", "scao"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    
    args = parser.parse_args()
    
    if args.test == "stress":
        run_stress_test(args.optimizer)
    else:
        run_convergence_test(args.optimizer, args.steps)

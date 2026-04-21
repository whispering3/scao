"""
inference.py — Load your SCAO-trained LoRA checkpoint and generate text

After running train_local.py, use this script to verify that the model
actually learned something useful. It loads the base GPT-2 model, injects
your LoRA weights from the saved checkpoint, and generates a completion
for a prompt of your choice.

Requirements:
    pip install transformers peft torch

Run:
    python inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    print("🚀 Loading base model and tokenizer...")
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the original GPT-2 weights — the LoRA adapter sits on top of these
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    print("🧠 Injecting your trained LoRA weights...")
    # Point this at the checkpoint folder created by train_local.py.
    # Hugging Face saves checkpoints as checkpoint-N where N is the step number.
    model = PeftModel.from_pretrained(base_model, "./resultado_scao_local/checkpoint-50")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # The prompt we used to validate that the model learned real-world context
    prompt = "The secret to a good software architecture is"

    print(f"\n✍️  Generating completion for: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print(result)
    print("=" * 50)
    # Expected output includes phrases like "its openness" — proof the model
    # absorbed real software architecture context during fine-tuning.


if __name__ == "__main__":
    main()
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ---- Login to Hugging Face Hub ----
# login("")  # Replace with your actual HF token

def merge_lora_and_push(
        model_path="health-agent-finetuned (1)/health-agent-finetuned/checkpoint-best",
        base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        hf_repo_id="aldsouza/health-agent-merged"
):
    """
    1. Load base model in FP16
    2. Merge LoRA fine-tuned weights
    3. Push merged FP16 model to Hugging Face Hub
    """

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load base model in FP16 (no BitsAndBytes) ----
    print("Loading base model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # ---- Load LoRA fine-tuned weights and merge ----
    print("Merging LoRA weights into base model...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # merge LoRA into base weights

    # ---- Push merged model & tokenizer to HF ----
    print(f"Pushing merged model to Hugging Face Hub: {hf_repo_id}...")
    model.push_to_hub(hf_repo_id, private=False)
    tokenizer.push_to_hub(hf_repo_id, private=False)

    print("✅ LoRA weights merged and model pushed successfully!")

if __name__ == "__main__":
    merge_lora_and_push()

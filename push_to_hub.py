import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from huggingface_hub import login

login("")

def push_finetuned_model_to_hf(
        model_path="health-agent-finetuned (1)/health-agent-finetuned/checkpoint-best",
        base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        hf_repo_id="your-username/health-agent-merged", merge=False
):
    """Merge PEFT model with base model and push to Hugging Face Hub."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA fine-tuned weights
    model = PeftModel.from_pretrained(base_model, model_path)

    # Merge LoRA into base model weights
    print("Merging LoRA weights into base model...")
    if merge:
        model = model.merge_and_unload()

    # Push merged model & tokenizer to HF Hub
    print(f"Pushing merged model to Hugging Face Hub: {hf_repo_id}")
    model.push_to_hub(hf_repo_id, private=False)  # private=True if you want private repo
    tokenizer.push_to_hub(hf_repo_id, private=False)

    print("✅ Model pushed successfully!")

if __name__ == "__main__":
    push_finetuned_model_to_hf(
        model_path="agent/health-agent-finetuned/checkpoint-best",
        base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        hf_repo_id="aldsouza/health-agent-rkllm-compatible",merge=True
    )

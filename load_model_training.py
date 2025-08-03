import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_finetuned_model_for_training(model_path="./health-agent-finetuned", base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load the fine-tuned model for continued training."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # QLoRA config for loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load fine-tuned model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.train()  # Set to training mode
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    print("Loading fine-tuned model for training...")
    model, tokenizer = load_finetuned_model_for_training()
    
    # Check trainable parameters
    model.print_trainable_parameters()
    
    print("Model loaded successfully for training!") 
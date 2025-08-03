import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_finetuned_model(model_path="./health-agent-finetuned/health-agent-finetuned/checkpoint-10", base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load the fine-tuned model for inference."""
    
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
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=4096, temperature=0.7):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3600).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0])
    return response

# Example usage
if __name__ == "__main__":
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    
    # Test the model
    prompt = "You are a medical AI assistant. Your job is to help users by understanding their request, identifying missing information if any, and then generating a structured plan using available tools.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n[\n  {\n    \"name\": \"request_missing_argument\",\n    \"description\": \"Ask the user for a required argument to complete a task\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"missing_field\": {\n          \"type\": \"string\",\n          \"description\": \"The specific field that is missing (e.g., 'patient_email', 'doctor_name', 'appointment_time', 'patient_id', 'appointment_id')\"\n        }\n      },\n      \"required\": [\n        \"missing_field\"\n      ]\n    }\n  },\n  {\n    \"name\": \"book_appointment\",\n    \"description\": \"Schedule an appointment with a doctor using doctor name and time\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"doctor_name\": {\n          \"type\": \"string\",\n          \"description\": \"Full name of the doctor (e.g., 'Dr. Smith', 'Dr. Sarah Johnson')\"\n        },\n        \"appointment_time\": {\n          \"type\": \"string\",\n          \"description\": \"Time for appointment (e.g., '10:00 AM Monday', 'next available', 'ASAP', '2:30 PM tomorrow')\"\n        }\n      },\n      \"required\": [\n        \"doctor_name\",\n        \"appointment_time\"\n      ]\n    }\n  }\n]\n</tools>\n\nFor each function call, use the following JSON format within your response:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"param_name\": \"param_value\"}}\n</tool_call>\n\nAvailable functions:\n- request_missing_argument: Ask for missing information needed to complete a task\n- book_appointment: Schedule appointments with doctors\n<|User|>I need to see a orthopedic surgeon as soon as possible.<|Assistant|>"

    print("Generating response...")
    response = generate_response(model, tokenizer, prompt)
    print(f"Generated response: {response}") 
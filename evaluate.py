import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, base_model_name):
    """Load the fine-tuned model and tokenizer."""
    
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
    model.eval()
    
    return model, tokenizer

def extract_user_query(text):
    """Extract the user query from the conversation text."""
    # Look for the user query pattern
    if "<|User|>" in text and "<|Assistant|>" in text:
        user_start = text.find("<|User|>") + len("<|User|>")
        assistant_start = text.find("<|Assistant|>")
        user_query = text[user_start:assistant_start].strip()
        return user_query
    return None

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def evaluate_model(model, tokenizer, test_dataset):
    """Evaluate the model on the test dataset."""
    
    results = []
    
    for i, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
        # Clean the text
        text = example["text"].replace("\uff5c", "|").replace("\u2581", "_")
        
        # Extract user query
        user_query = extract_user_query(text)
        if not user_query:
            logger.warning(f"Could not extract user query from example {i}")
            continue
        
        # Create prompt (everything up to the assistant response)
        assistant_start = text.find("<|Assistant|>")
        if assistant_start == -1:
            logger.warning(f"Could not find assistant marker in example {i}")
            continue
            
        prompt = text[:assistant_start]
        
        # Get the expected response (everything after assistant marker)
        expected_response = text[assistant_start + len("<|Assistant|>"):].strip()
        
        # Generate response
        try:
            generated_response = generate_response(model, tokenizer, prompt)
            
            result = {
                "example_id": i,
                "user_query": user_query,
                "expected_response": expected_response,
                "generated_response": generated_response,
                "prompt_length": len(tokenizer.encode(prompt)),
                "expected_length": len(tokenizer.encode(expected_response)),
                "generated_length": len(tokenizer.encode(generated_response)),
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error generating response for example {i}: {e}")
            continue
    
    return results

def calculate_metrics(results):
    """Calculate evaluation metrics."""
    
    if not results:
        return {}
    
    # Length metrics
    expected_lengths = [r["expected_length"] for r in results]
    generated_lengths = [r["generated_length"] for r in results]
    
    metrics = {
        "num_examples": len(results),
        "avg_expected_length": np.mean(expected_lengths),
        "avg_generated_length": np.mean(generated_lengths),
        "length_ratio": np.mean(generated_lengths) / np.mean(expected_lengths) if np.mean(expected_lengths) > 0 else 0,
    }
    
    return metrics

def save_results(results, metrics, output_file="evaluation_results.json"):
    """Save evaluation results to a JSON file."""
    
    output = {
        "metrics": metrics,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def print_summary(metrics, results):
    """Print a summary of the evaluation results."""
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"Number of examples evaluated: {metrics['num_examples']}")
    print(f"Average expected response length: {metrics['avg_expected_length']:.1f} tokens")
    print(f"Average generated response length: {metrics['avg_generated_length']:.1f} tokens")
    print(f"Length ratio (generated/expected): {metrics['length_ratio']:.2f}")
    
    print("\n" + "="*50)
    print("SAMPLE RESPONSES")
    print("="*50)
    
    for i, result in enumerate(results[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"User Query: {result['user_query']}")
        print(f"Expected: {result['expected_response'][:200]}...")
        print(f"Generated: {result['generated_response'][:200]}...")
        print("-" * 30)

def main():
    # Configuration
    model_path = "./health-agent-finetuned"  # Path to your fine-tuned model
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    test_data_file = "tokenized_test_data.jsonl"
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path, base_model_name)
    
    print("Loading test dataset...")
    test_dataset = load_dataset("json", data_files=test_data_file, split="train")
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("Evaluating model...")
    results = evaluate_model(model, tokenizer, test_dataset)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    
    print("Saving results...")
    save_results(results, metrics)
    
    print("Printing summary...")
    print_summary(metrics, results)

if __name__ == "__main__":
    main() 
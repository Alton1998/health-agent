import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
import torch

def analyze_token_lengths():
    """Analyze token lengths in the dataset to determine optimal max_length"""
    
    print("🔍 Analyzing token lengths in your dataset...")
    
    # Load tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_from_disk("./health-agent-dataset")
    
    # Sample size for analysis (use all data if small, sample if large)
    sample_size = min(1000, len(dataset["train"]))
    print(f"📊 Analyzing {sample_size} samples from {len(dataset['train'])} total samples")
    
    # Tokenize samples
    token_lengths = []
    formatted_lengths = []
    
    for i, example in enumerate(dataset["train"]):
        if i >= sample_size:
            break
            
        try:
            # Get original messages
            messages = example["messages"]
            
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Tokenize
            tokenized = tokenizer(
                formatted_text,
                truncation=False,
                padding=False,
                return_tensors=None
            )
            
            length = len(tokenized["input_ids"])
            token_lengths.append(length)
            formatted_lengths.append(len(formatted_text))
            
        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")
            continue
    
    if not token_lengths:
        print("❌ No valid samples found!")
        return
    
    # Calculate statistics
    token_lengths = np.array(token_lengths)
    formatted_lengths = np.array(formatted_lengths)
    
    print("\n📈 TOKEN LENGTH STATISTICS:")
    print(f"Total samples analyzed: {len(token_lengths)}")
    print(f"Mean token length: {np.mean(token_lengths):.1f}")
    print(f"Median token length: {np.median(token_lengths):.1f}")
    print(f"Standard deviation: {np.std(token_lengths):.1f}")
    print(f"Min token length: {np.min(token_lengths)}")
    print(f"Max token length: {np.max(token_lengths)}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    print(f"\n📊 PERCENTILES:")
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        print(f"{p}th percentile: {value:.0f} tokens")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Conservative (covers 90% of data)
    conservative = int(np.percentile(token_lengths, 90))
    print(f"Conservative (90% coverage): {conservative} tokens")
    
    # Balanced (covers 95% of data)
    balanced = int(np.percentile(token_lengths, 95))
    print(f"Balanced (95% coverage): {balanced} tokens")
    
    # Aggressive (covers 99% of data)
    aggressive = int(np.percentile(token_lengths, 99))
    print(f"Aggressive (99% coverage): {aggressive} tokens")
    
    # Memory considerations
    print(f"\n💾 MEMORY CONSIDERATIONS:")
    print(f"With batch_size=1 and gradient_accumulation_steps=4:")
    print(f"  - Conservative: ~{conservative * 4} tokens per effective batch")
    print(f"  - Balanced: ~{balanced * 4} tokens per effective batch")
    print(f"  - Aggressive: ~{aggressive * 4} tokens per effective batch")
    
    # Plot distribution
    try:
        plt.figure(figsize=(12, 8))
        
        # Token length distribution
        plt.subplot(2, 2, 1)
        plt.hist(token_lengths, bins=50, alpha=0.7, color='blue')
        plt.axvline(conservative, color='orange', linestyle='--', label=f'90% ({conservative})')
        plt.axvline(balanced, color='red', linestyle='--', label=f'95% ({balanced})')
        plt.axvline(aggressive, color='purple', linestyle='--', label=f'99% ({aggressive})')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title('Token Length Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 2)
        sorted_lengths = np.sort(token_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        plt.plot(sorted_lengths, cumulative, 'b-', linewidth=2)
        plt.axhline(90, color='orange', linestyle='--', alpha=0.7)
        plt.axhline(95, color='red', linestyle='--', alpha=0.7)
        plt.axhline(99, color='purple', linestyle='--', alpha=0.7)
        plt.xlabel('Token Length')
        plt.ylabel('Cumulative %')
        plt.title('Cumulative Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(token_lengths)
        plt.ylabel('Token Length')
        plt.title('Token Length Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Text length vs token length
        plt.subplot(2, 2, 4)
        plt.scatter(formatted_lengths, token_lengths, alpha=0.5, s=1)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Token Length')
        plt.title('Text vs Token Length')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('token_length_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved as 'token_length_analysis.png'")
        
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
    
    # Final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print(f"Set tokenizer.model_max_length = {balanced}")
    print(f"This covers {np.sum(token_lengths <= balanced) / len(token_lengths) * 100:.1f}% of your data")
    
    # Data loss analysis
    data_loss_90 = (1 - np.sum(token_lengths <= conservative) / len(token_lengths)) * 100
    data_loss_95 = (1 - np.sum(token_lengths <= balanced) / len(token_lengths)) * 100
    data_loss_99 = (1 - np.sum(token_lengths <= aggressive) / len(token_lengths)) * 100
    
    print(f"\n📉 DATA LOSS ANALYSIS:")
    print(f"Conservative ({conservative}): {data_loss_90:.1f}% of data would be truncated")
    print(f"Balanced ({balanced}): {data_loss_95:.1f}% of data would be truncated")
    print(f"Aggressive ({aggressive}): {data_loss_99:.1f}% of data would be truncated")

if __name__ == "__main__":
    analyze_token_lengths() 
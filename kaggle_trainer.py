import logging
import torch
import math
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# KAGGLE-OPTIMIZED CONFIGURATION
# =============================================================================

# Model and Dataset Configuration
MODEL_NAME = "aldsouza/health-agent-lora-rkllm-compatible"
DATASET_NAME = "aldsouza/health-agent-dataset"
OUTPUT_DIR = "./health-agent-kaggle-trained"

# Training Configuration (Optimized for Kaggle's T4 GPU)
NUM_EPOCHS = 3  # Reduced for Kaggle time limits
BATCH_SIZE = 1  # Conservative for T4 GPU
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4  # Reduced for faster training
MAX_GRAD_NORM = 0.3
MAX_LENGTH = 512  # Reduced for memory efficiency

# Training Control
EVAL_STEPS = 25  # More frequent evaluation
SAVE_STEPS = 25  # More frequent saving
LOG_STEPS = 1

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

def tokenize_function(example, tokenizer):
    """Tokenize a single example from the dataset."""
    try:
        messages = example["messages"]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.copy()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    except Exception as e:
        print(f"Error tokenizing example: {e}")
        return {
            "input_ids": [tokenizer.eos_token_id],
            "attention_mask": [1],
            "labels": [tokenizer.eos_token_id]
        }

def custom_collate_fn(batch, tokenizer):
    """Custom collate function for batching data."""
    max_length = max(len(item["input_ids"]) for item in batch)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]
        
        # Convert to tensors if they're lists
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Calculate padding length
        pad_length = max_length - len(input_ids)
        
        # Pad with pad_token_id
        padded_input_ids.append(torch.cat([input_ids, torch.full((pad_length,), tokenizer.pad_token_id, dtype=input_ids.dtype)]))
        padded_attention_mask.append(torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)]))
        padded_labels.append(torch.cat([labels, torch.full((pad_length,), -100, dtype=labels.dtype)]))
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels)
    }

def print_memory_usage(stage=""):
    """Print GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        print(f"🔍 GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

def train_kaggle():
    """
    Kaggle-optimized training function.
    """
    print("🚀 Starting Kaggle Training")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 GPU: {gpu_name}")
        print(f"🔍 GPU Memory: {gpu_memory:.1f}GB")
    
    # --- Model & Tokenizer Setup ---
    print(f"🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LENGTH
    
    # --- Dataset Loading from Hub ---
    print(f"📊 Loading dataset from Hub: {DATASET_NAME}")
    
    try:
        dataset = load_dataset(DATASET_NAME)
        print(f"✅ Successfully loaded dataset from Hub")
        print(f"📋 Dataset info: {dataset}")
    except Exception as e:
        print(f"❌ Error loading dataset from Hub: {e}")
        print("💡 Make sure the dataset exists and you have access to it")
        return
    
    # Split the dataset (smaller for Kaggle)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)  # Smaller test set
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    # Limit dataset size for Kaggle (optional)
    if len(train_dataset) > 10000:  # Limit to 10k samples for faster training
        train_dataset = train_dataset.select(range(10000))
        print(f"📊 Limited train dataset to 10,000 samples for Kaggle")
    
    print(f"   - Train dataset size: {len(train_dataset)}")
    print(f"   - Test dataset size: {len(test_dataset)}")
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        remove_columns=test_dataset.column_names
    )
    
    # --- Model Setup ---
    print("🤖 Loading pre-trained LoRA model...")
    
    # Load model without quantization (model is already optimized)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,  # We'll move to device manually
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()
    
    # Model already has LoRA adapters, so we use it directly
    peft_model = model
    
    # Check if model has trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable %: {100 * trainable_params / total_params:.2f}%")
    print_memory_usage("after model loading")
    
    # Move model to device
    peft_model = peft_model.to(device)
    print_memory_usage("after moving to device")
    
    # --- DataLoaders ---
    print("📦 Setting up data loaders...")
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=0,  # Set to 0 for Kaggle compatibility
        pin_memory=True,
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        test_dataset.select(range(min(100, len(test_dataset)))),  # Smaller eval set
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=0,
        pin_memory=True,
    )
    
    # --- Training Setup ---
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training state
    global_step = 0
    best_eval_loss = float('inf')
    
    # --- Training Loop ---
    print("🚀 Starting training...")
    print(f"📊 Training configuration:")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Dataset: {DATASET_NAME}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Max length: {MAX_LENGTH}")
    print_memory_usage("before training")
    
    for epoch in range(NUM_EPOCHS):
        peft_model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in progress:
            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Clear cache periodically to free memory
            if step % 5 == 0:
                torch.cuda.empty_cache()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"⚠️ Skipping NaN loss at step {step}")
                continue
            
            # Scale loss for gradient accumulation
            scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
            
            # Update weights
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), MAX_GRAD_NORM)
                
                if not torch.isnan(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    print(f"⚠️ Skipping optimizer step due to NaN grad_norm at step {step}")
                
                global_step += 1
            
            # Accumulate losses
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if global_step % LOG_STEPS == 0:
                avg_loss = running_loss / LOG_STEPS
                progress.set_postfix(loss=avg_loss)
                running_loss = 0.0
            
            # --- Evaluation ---
            if global_step % EVAL_STEPS == 0 and global_step > 0:
                peft_model.eval()
                eval_losses = []
                
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        ids = eval_batch["input_ids"].to(device, non_blocking=True)
                        mask = eval_batch["attention_mask"].to(device, non_blocking=True)
                        labels = eval_batch["labels"].to(device, non_blocking=True)
                        
                        with torch.cuda.amp.autocast():
                            outputs = peft_model(input_ids=ids, attention_mask=mask, labels=labels)
                            eval_loss = outputs.loss
                        
                        if not torch.isnan(eval_loss):
                            eval_losses.append(eval_loss.item())
                
                if eval_losses:
                    mean_eval_loss = sum(eval_losses) / len(eval_losses)
                    print(f"✅ Eval at step {global_step} | loss: {mean_eval_loss:.4f}")
                    print_memory_usage(f"after eval step {global_step}")
                    
                    # Save best model if eval loss improved
                    if mean_eval_loss < best_eval_loss:
                        best_eval_loss = mean_eval_loss
                        save_path = f"{OUTPUT_DIR}/checkpoint-best"
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Save model
                        peft_model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        
                        # Save training info
                        training_info = {
                            'model_name': MODEL_NAME,
                            'dataset_name': DATASET_NAME,
                            'global_step': global_step,
                            'best_eval_loss': float(best_eval_loss),
                            'batch_size': BATCH_SIZE,
                            'learning_rate': LEARNING_RATE,
                            'kaggle_training': True
                        }
                        with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
                            json.dump(training_info, f, indent=2)
                        
                        print(f"💾 New best model saved at step {global_step} with loss: {best_eval_loss:.4f}")
                
                peft_model.train()
            
            # --- Save Checkpoint ---
            if global_step % SAVE_STEPS == 0 and global_step > 0:
                save_path = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                os.makedirs(save_path, exist_ok=True)
                peft_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                
                # Save training info
                training_info = {
                    'model_name': MODEL_NAME,
                    'dataset_name': DATASET_NAME,
                    'global_step': global_step,
                    'best_eval_loss': float(best_eval_loss) if best_eval_loss != float('inf') else 'inf',
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'epoch': epoch,
                    'kaggle_training': True
                }
                with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
                    json.dump(training_info, f, indent=2)
                
                print(f"💾 Checkpoint saved at step {global_step}")
                print_memory_usage(f"after saving checkpoint {global_step}")
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"📊 Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
        print_memory_usage(f"end of epoch {epoch}")
    
    # --- Final Save ---
    print("💾 Saving final model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save final training info
    final_training_info = {
        'model_name': MODEL_NAME,
        'dataset_name': DATASET_NAME,
        'final_epoch': epoch,
        'final_step': global_step,
        'best_eval_loss': float(best_eval_loss) if best_eval_loss != float('inf') else 'inf',
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'training_completed': True,
        'kaggle_training': True
    }
    with open(os.path.join(OUTPUT_DIR, 'final_training_info.json'), 'w') as f:
        json.dump(final_training_info, f, indent=2)
    
    print(f"✅ Final model saved to {OUTPUT_DIR}")
    print(f"📊 Training completed with dataset: {DATASET_NAME}")
    print(f"🎯 Best evaluation loss: {best_eval_loss:.4f}")
    print_memory_usage("final save")
    
    # Show how to download the model
    print("\n" + "="*50)
    print("📥 To download your trained model:")
    print("1. Go to the 'Output' tab in Kaggle")
    print("2. Find the 'health-agent-kaggle-trained' folder")
    print("3. Download the entire folder")
    print("4. Use the model in your local environment")
    print("="*50)

def main():
    """Main function to launch Kaggle training."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Training requires GPUs.")
        print("💡 Make sure you have enabled GPU in Kaggle settings")
        return
    
    # Check GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🔍 GPU: {gpu_name}")
    print(f"🔍 GPU Memory: {gpu_memory:.1f}GB")
    
    if gpu_memory < 10:
        print("⚠️ GPU has less than 10GB memory. Training might be slow or fail.")
        print("💡 Consider reducing batch size or sequence length")
    
    print(f"🚀 Starting Kaggle Training")
    print(f"📊 Dataset: {DATASET_NAME}")
    print(f"📊 Model: {MODEL_NAME}")
    print(f"📊 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"📊 Note: Using pre-trained LoRA model (no quantization needed)")
    
    # Run training
    train_kaggle()

if __name__ == "__main__":
    main()

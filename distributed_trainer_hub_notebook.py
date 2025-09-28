import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
import os
import json
from tqdm import tqdm
from datasets import load_dataset
# from peft import LoraConfig, get_peft_model  # Not needed since model already has LoRA
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# HARDCODED CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# =============================================================================

# Model and Dataset Configuration
MODEL_NAME = "aldsouza/health-agent-lora-rkllm-compatible"
DATASET_NAME = "aldsouza/health-agent-dataset"
OUTPUT_DIR = "./health-agent-distributed-hub"

# Training Configuration (Conservative for stability)
NUM_EPOCHS = 2  # Reduced for testing
BATCH_SIZE = 1  # Per GPU
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 2  # Further reduced for stability
MAX_GRAD_NORM = 0.3
MAX_LENGTH = 256  # Much smaller for stability

# LoRA Configuration (Model already has LoRA adapters)
# LORA_R = 8
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.1

# Training Control
EVAL_STEPS = 50
SAVE_STEPS = 50
LOG_STEPS = 1

# Distributed Training
WORLD_SIZE = 2  # Number of GPUs

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

def setup_distributed(rank, world_size, backend='nccl'):
    """
    Initialize the distributed training environment.
    
    Args:
        rank: The rank of the current process (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    print(f"🚀 Process {rank}/{world_size} initialized on GPU {rank}")

def cleanup_distributed():
    """Clean up the distributed training environment."""
    dist.destroy_process_group()

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

def print_memory_usage(rank, stage=""):
    """Print GPU memory usage for debugging."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔍 Memory Usage {stage}:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.1f}GB total")

def train_single_gpu():
    """
    Single GPU training function for notebook environments.
    """
    print("🚀 Starting single GPU training (Notebook Compatible)")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Device setup
    # Use device_map to distribute model across available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔍 Detected {num_gpus} GPUs")
        if num_gpus > 1:
            # Use device_map to distribute model across GPUs
            device_map = "auto"  # Let transformers handle GPU distribution
            device = torch.device('cuda:0')  # Primary device for training
        else:
            device_map = None
            device = torch.device('cuda:0')
    else:
        device_map = None
        device = torch.device('cpu')
    print(f"🔧 Using device: {device}")
    
    # --- Model & Tokenizer Setup ---
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
    
    # Split the dataset
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    # Limit dataset size for stability and testing
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))  # Much smaller
    test_dataset = test_dataset.select(range(min(200, len(test_dataset))))      # Much smaller
    
    print(f"   - Train dataset size: {len(train_dataset)} (limited for memory)")
    print(f"   - Test dataset size: {len(test_dataset)} (limited for memory)")
    
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
        device_map=device_map,  # Use device_map for multi-GPU distribution
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
    print_memory_usage(0, "after model loading")
    
    # Move model to device
    # Model is already on the correct device(s) via device_map
    # peft_model = peft_model.to(device)  # Not needed with device_map
    print_memory_usage(0, "after moving to device")
    
    # --- DataLoaders ---
    print("📦 Setting up data loaders...")
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=1,
        pin_memory=True,
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        test_dataset.select(range(min(500, len(test_dataset)))),
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=1,
        pin_memory=True,
    )
    
    # --- Training Setup ---
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)
    # Note: No scaler needed for FP16 without mixed precision
    
    # Training state
    global_step = 0
    best_eval_loss = float('inf')
    
    # --- Training Loop ---
    print("🚀 Starting training...")
    print_memory_usage(0, "before training")
    
    # Add GPU health check
    try:
        # Test GPU with a simple operation
        test_tensor = torch.randn(10, 10).to(device)
        test_result = torch.mm(test_tensor, test_tensor.t())
        del test_tensor, test_result
        torch.cuda.empty_cache()
        print("✅ GPU health check passed")
    except Exception as e:
        print(f"❌ GPU health check failed: {e}")
        return
    
    for epoch in range(NUM_EPOCHS):
        peft_model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in progress:
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                # Clear cache more frequently for stability
                if step % 5 == 0:
                    torch.cuda.empty_cache()
                
                # Forward pass (no mixed precision to avoid gradient scaling issues)
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss for gradient accumulation
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Skipping invalid loss at step {step}: {loss.item()}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), MAX_GRAD_NORM)
                    
                    if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                    else:
                        print(f"⚠️ Skipping optimizer step due to invalid grad_norm at step {step}: {grad_norm}")
                        optimizer.zero_grad()  # Clear gradients even if we skip the step
                
                # Update progress bar
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'step': f'{global_step}',
                    'epoch': f'{epoch}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ OOM at step {step}: {e}")
                    torch.cuda.empty_cache()
                    break
                else:
                    print(f"❌ Runtime error at step {step}: {e}")
                    break
            except Exception as e:
                print(f"❌ Unexpected error at step {step}: {e}")
                break
            
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
                        
                        outputs = peft_model(input_ids=ids, attention_mask=mask, labels=labels)
                        eval_loss = outputs.loss
                        
                        if not torch.isnan(eval_loss):
                            eval_losses.append(eval_loss.item())
                
                if eval_losses:
                    mean_eval_loss = sum(eval_losses) / len(eval_losses)
                    print(f"✅ Eval at step {global_step} | loss: {mean_eval_loss:.4f}")
                    print_memory_usage(0, f"after eval step {global_step}")
                    
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
                            'world_size': 1,
                            'batch_size': BATCH_SIZE,
                            'learning_rate': LEARNING_RATE
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
                    'world_size': 1,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'epoch': epoch
                }
                with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
                    json.dump(training_info, f, indent=2)
                
                print(f"💾 Checkpoint saved at step {global_step}")
                print_memory_usage(0, f"after saving checkpoint {global_step}")
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"📊 Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
        print_memory_usage(0, f"end of epoch {epoch}")
    
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
        'world_size': 1,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'training_completed': True
    }
    with open(os.path.join(OUTPUT_DIR, 'final_training_info.json'), 'w') as f:
        json.dump(final_training_info, f, indent=2)
    
    print(f"✅ Final model saved to {OUTPUT_DIR}")
    print(f"📊 Training completed with dataset: {DATASET_NAME}")
    print(f"🎯 Best evaluation loss: {best_eval_loss:.4f}")
    print_memory_usage(0, "final save")

def main():
    """Main function to launch training."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Training requires GPUs.")
        return
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"🔍 Available GPUs: {available_gpus}")
    
    # Check GPU memory
    for i in range(available_gpus):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"🔍 GPU {i}: {gpu_memory:.1f}GB total memory")
    
    print(f"🚀 Starting training (Notebook Compatible)")
    print(f"📊 Dataset: {DATASET_NAME}")
    print(f"📊 Model: {MODEL_NAME}")
    print(f"📊 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"📊 Note: Using pre-trained LoRA model (no quantization needed)")
    
    # Run single GPU training (notebook compatible)
    train_single_gpu()

if __name__ == "__main__":
    main()

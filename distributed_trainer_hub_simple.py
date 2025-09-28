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

# Training Configuration (Optimized for 2x 16GB GPUs)
NUM_EPOCHS = 5
BATCH_SIZE = 1  # Per GPU
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 0.3
MAX_LENGTH = 1024  # Reduced for memory efficiency

# LoRA Configuration (Model already has LoRA adapters)
# LORA_R = 8
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.1

# Training Control
EVAL_STEPS = 50
SAVE_STEPS = 50
LOG_STEPS = 1

# Distributed Training
WORLD_SIZE = 2  # Number of GPUs to use

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
        allocated = torch.cuda.memory_allocated(rank) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(rank) / 1024**3    # GB
        print(f"🔍 GPU {rank} Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

def train_worker(rank, world_size):
    """
    Main training function for each distributed worker.
    
    Args:
        rank: The rank of the current process
        world_size: Total number of processes
    """
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42 + rank)  # Different seed for each process
    
    # Device setup
    device = torch.device(f'cuda:{rank}')
    
    # Only rank 0 should print setup information
    if rank == 0:
        print(f"🔧 Distributed Training Configuration (Hub Dataset):")
        print(f"   - Model: {MODEL_NAME}")
        print(f"   - Dataset: {DATASET_NAME}")
        print(f"   - World Size: {world_size}")
        print(f"   - Batch Size per GPU: {BATCH_SIZE}")
        print(f"   - Effective Batch Size: {BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS}")
        print(f"   - Learning Rate: {LEARNING_RATE}")
        print(f"   - Epochs: {NUM_EPOCHS}")
        print(f"   - Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"   - Max Sequence Length: {MAX_LENGTH}")
    
    # --- Model & Tokenizer Setup ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LENGTH
    
    # --- Dataset Loading from Hub ---
    if rank == 0:
        print(f"📊 Loading dataset from Hub: {DATASET_NAME}")
    
    try:
        dataset = load_dataset(DATASET_NAME)
        if rank == 0:
            print(f"✅ Successfully loaded dataset from Hub")
            print(f"📋 Dataset info: {dataset}")
    except Exception as e:
        if rank == 0:
            print(f"❌ Error loading dataset from Hub: {e}")
            print("💡 Make sure the dataset exists and you have access to it")
        cleanup_distributed()
        return
    
    # Split the dataset
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    if rank == 0:
        print(f"   - Train dataset size: {len(train_dataset)}")
        print(f"   - Test dataset size: {len(test_dataset)}")
    
    # Tokenize datasets
    if rank == 0:
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
    if rank == 0:
        print("🤖 Loading pre-trained LoRA model...")
    
    # Load model without quantization (model is already optimized)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,  # We'll move to device manually
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 to avoid gradient scaling issues
        low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()
    
    # Model already has LoRA adapters, so we use it directly
    peft_model = model
    
    if rank == 0:
        # Check if model has trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable %: {100 * trainable_params / total_params:.2f}%")
        print_memory_usage(rank, "after model loading")
    
    # Move model to device
    peft_model = peft_model.to(device)
    
    # Wrap model with DDP
    peft_model = DDP(peft_model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        print_memory_usage(rank, "after DDP wrapping")
    
    # --- DataLoaders with Distributed Sampling ---
    if rank == 0:
        print("📦 Setting up data loaders...")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    eval_sampler = DistributedSampler(
        test_dataset.select(range(min(500, len(test_dataset)))), 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders with memory optimizations
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=1,  # Reduced to save memory
        pin_memory=True,
        persistent_workers=False,  # Don't keep workers alive
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        test_dataset.select(range(min(500, len(test_dataset)))),
        batch_size=BATCH_SIZE,
        sampler=eval_sampler,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
    )
    
    # --- Training Setup ---
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    # Training state
    global_step = 0
    best_eval_loss = float('inf')
    
    # --- Training Loop ---
    if rank == 0:
        print("🚀 Starting distributed training...")
        print_memory_usage(rank, "before training")
    
    for epoch in range(NUM_EPOCHS):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        peft_model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar only on rank 0
        if rank == 0:
            progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        else:
            progress = enumerate(train_dataloader)
        
        for step, batch in progress:
            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Clear cache periodically to free memory
            if step % 10 == 0:
                torch.cuda.empty_cache()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            if torch.isnan(loss):
                if rank == 0:
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
                    if rank == 0:
                        print(f"⚠️ Skipping optimizer step due to NaN grad_norm at step {step}")
                
                global_step += 1
            
            # Accumulate losses
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar (only on rank 0)
            if rank == 0 and global_step % LOG_STEPS == 0:
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
                        
                        with torch.amp.autocast('cuda'):
                            outputs = peft_model(input_ids=ids, attention_mask=mask, labels=labels)
                            eval_loss = outputs.loss
                        
                        if not torch.isnan(eval_loss):
                            eval_losses.append(eval_loss.item())
                
                if eval_losses:
                    # Gather evaluation losses from all processes
                    eval_losses_tensor = torch.tensor(eval_losses, device=device)
                    gathered_losses = [torch.zeros_like(eval_losses_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered_losses, eval_losses_tensor)
                    
                    # Calculate mean across all processes
                    all_losses = torch.cat(gathered_losses).cpu().numpy()
                    mean_eval_loss = all_losses.mean()
                    
                    if rank == 0:
                        print(f"✅ Eval at step {global_step} | loss: {mean_eval_loss:.4f}")
                        print_memory_usage(rank, f"after eval step {global_step}")
                        
                        # Save best model if eval loss improved
                        if mean_eval_loss < best_eval_loss:
                            best_eval_loss = mean_eval_loss
                            save_path = f"{OUTPUT_DIR}/checkpoint-best"
                            
                            # Save model (only rank 0)
                            peft_model.module.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            
                            # Save training info
                            training_info = {
                                'model_name': MODEL_NAME,
                                'dataset_name': DATASET_NAME,
                                'global_step': global_step,
                                'best_eval_loss': float(best_eval_loss),
                                'world_size': world_size,
                                'batch_size': BATCH_SIZE,
                                'learning_rate': LEARNING_RATE
                            }
                            with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
                                json.dump(training_info, f, indent=2)
                            
                            print(f"💾 New best model saved at step {global_step} with loss: {best_eval_loss:.4f}")
                
                peft_model.train()
            
            # --- Save Checkpoint ---
            if global_step % SAVE_STEPS == 0 and global_step > 0 and rank == 0:
                save_path = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                peft_model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                
                # Save training info
                training_info = {
                    'model_name': MODEL_NAME,
                    'dataset_name': DATASET_NAME,
                    'global_step': global_step,
                    'best_eval_loss': float(best_eval_loss) if best_eval_loss != float('inf') else 'inf',
                    'world_size': world_size,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'epoch': epoch
                }
                with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
                    json.dump(training_info, f, indent=2)
                
                print(f"💾 Checkpoint saved at step {global_step}")
                print_memory_usage(rank, f"after saving checkpoint {global_step}")
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        # Gather epoch losses from all processes
        epoch_loss_tensor = torch.tensor(avg_epoch_loss, device=device)
        gathered_epoch_losses = [torch.zeros_like(epoch_loss_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_epoch_losses, epoch_loss_tensor)
        
        # Calculate mean epoch loss across all processes
        all_epoch_losses = torch.cat(gathered_epoch_losses).cpu().numpy()
        mean_epoch_loss = all_epoch_losses.mean()
        
        if rank == 0:
            print(f"📊 Epoch {epoch} average loss: {mean_epoch_loss:.4f}")
            print_memory_usage(rank, f"end of epoch {epoch}")
    
    # --- Final Save ---
    if rank == 0:
        print("💾 Saving final model...")
        final_save_path = OUTPUT_DIR
        peft_model.module.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        # Save final training info
        final_training_info = {
            'model_name': MODEL_NAME,
            'dataset_name': DATASET_NAME,
            'final_epoch': epoch,
            'final_step': global_step,
            'best_eval_loss': float(best_eval_loss) if best_eval_loss != float('inf') else 'inf',
            'world_size': world_size,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'training_completed': True
        }
        with open(os.path.join(final_save_path, 'final_training_info.json'), 'w') as f:
            json.dump(final_training_info, f, indent=2)
        
        print(f"✅ Final model saved to {final_save_path}")
        print_memory_usage(rank, "final save")
    
    # Cleanup
    cleanup_distributed()

def main():
    """Main function to launch distributed training."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Distributed training requires GPUs.")
        return
    
    # Check if we have enough GPUs
    available_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {available_gpus} GPUs")
    
    if WORLD_SIZE > available_gpus:
        print(f"⚠️ Requested {WORLD_SIZE} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        actual_world_size = available_gpus
    else:
        actual_world_size = WORLD_SIZE
    
    # Force distributed training if we have 2+ GPUs
    if available_gpus >= 2 and actual_world_size == 1:
        print(f"🚀 Found {available_gpus} GPUs, forcing distributed training with 2 GPUs")
        actual_world_size = 2
    
    # Check GPU memory
    for i in range(actual_world_size):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"🔍 GPU {i}: {gpu_memory:.1f}GB total memory")
        if gpu_memory < 14:  # Leave some headroom
            print(f"⚠️ GPU {i} has less than 14GB memory. Training might fail.")
    
    print(f"🚀 Starting distributed training on {actual_world_size} GPUs (Pre-trained LoRA Model)")
    print(f"📊 Dataset: {DATASET_NAME}")
    print(f"📊 Model: {MODEL_NAME}")
    print(f"📊 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Effective batch size: {BATCH_SIZE * actual_world_size * GRADIENT_ACCUMULATION_STEPS}")
    print(f"📊 Note: Using pre-trained LoRA model (no quantization needed)")
    
    # Launch distributed training
    try:
        mp.spawn(
            train_worker,
            args=(actual_world_size,),
            nprocs=actual_world_size,
            join=True
        )
    except Exception as e:
        print(f"❌ Error with multiprocessing spawn: {e}")
        print("💡 This might be due to running in Jupyter notebook or interactive environment")
        print("🔄 Trying alternative approach...")
        
        # Alternative: Run single process if spawn fails
        if actual_world_size == 1:
            print("🚀 Running single GPU training...")
            train_worker(0, 1)
        else:
            print("❌ Multi-GPU training requires running as a script, not in Jupyter notebook")
            print("💡 Please run: python distributed_trainer_hub_simple.py")

if __name__ == "__main__":
    main()

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
# HARDCODED CONFIGURATION - FORCE 2 GPU DISTRIBUTED TRAINING
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

# Training Control
EVAL_STEPS = 50
SAVE_STEPS = 50
LOG_STEPS = 1

# FORCE 2 GPU DISTRIBUTED TRAINING
WORLD_SIZE = 2  # ALWAYS use 2 GPUs

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
    if dist.is_initialized():
        dist.destroy_process_group()

def print_memory_usage(rank, stage=""):
    """Print GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / 1024**3
        reserved = torch.cuda.memory_reserved(rank) / 1024**3
        print(f"🔍 GPU {rank} Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def train_worker(rank, world_size):
    """
    Training worker function for each GPU process.
    
    Args:
        rank: The rank of the current process (0 to world_size-1)
        world_size: Total number of processes
    """
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    print(f"🚀 Starting training on GPU {rank}")
    print_memory_usage(rank, "startup")
    
    # Load tokenizer
    print(f"📚 Loading tokenizer on GPU {rank}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LENGTH
    
    # Load dataset from Hugging Face Hub
    print(f"📊 Loading dataset from Hub on GPU {rank}...")
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset['train']
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Combine instruction and response
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Response:\n{examples['response'][i]}"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    print(f"🔄 Tokenizing dataset on GPU {rank}...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing"
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    print_memory_usage(rank, "after data loading")
    
    # Load model
    print(f"🤖 Loading model on GPU {rank}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,  # We'll move to device manually
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 to avoid gradient scaling issues
        low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
    )
    
    # Set pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Move model to device
    model = model.to(device)
    
    # Model already has LoRA adapters, so we use it directly
    peft_model = model
    
    # Wrap model with DDP
    peft_model = DDP(peft_model, device_ids=[rank], output_device=rank)
    
    print_memory_usage(rank, "after model loading")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        peft_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop
    peft_model.train()
    total_steps = len(dataloader) * NUM_EPOCHS
    global_step = 0
    accumulated_loss = 0.0
    
    print(f"🚀 Starting training on GPU {rank} for {NUM_EPOCHS} epochs")
    print(f"📊 Total steps: {total_steps}")
    print(f"📊 Steps per epoch: {len(dataloader)}")
    print(f"📊 Effective batch size: {BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS}")
    
    for epoch in range(NUM_EPOCHS):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (GPU {rank})", disable=rank != 0)
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = peft_model(**batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            scaler.scale(loss).backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), MAX_GRAD_NORM)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging (only on rank 0)
                if rank == 0 and global_step % LOG_STEPS == 0:
                    avg_loss = accumulated_loss / GRADIENT_ACCUMULATION_STEPS
                    print(f"Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}")
                    accumulated_loss = 0.0
                
                # Save checkpoint (only on rank 0)
                if rank == 0 and global_step % SAVE_STEPS == 0:
                    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save model state dict (unwrap from DDP)
                    model_to_save = peft_model.module if hasattr(peft_model, 'module') else peft_model
                    model_to_save.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save training info
                    training_info = {
                        'global_step': global_step,
                        'epoch': epoch,
                        'loss': avg_loss,
                        'learning_rate': LEARNING_RATE,
                        'model_name': MODEL_NAME,
                        'dataset_name': DATASET_NAME
                    }
                    with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
                        json.dump(training_info, f, indent=2)
                    
                    print(f"💾 Checkpoint saved at step {global_step}")
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        # Print epoch summary (only on rank 0)
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"📊 Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
            print_memory_usage(rank, f"epoch {epoch+1} end")
    
    # Final save (only on rank 0)
    if rank == 0:
        final_save_path = os.path.join(OUTPUT_DIR, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        
        # Save final model
        model_to_save = peft_model.module if hasattr(peft_model, 'module') else peft_model
        model_to_save.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        # Save final training info
        final_training_info = {
            'total_steps': global_step,
            'total_epochs': NUM_EPOCHS,
            'final_loss': accumulated_loss / GRADIENT_ACCUMULATION_STEPS if accumulated_loss > 0 else 0,
            'model_name': MODEL_NAME,
            'dataset_name': DATASET_NAME,
            'training_completed': True
        }
        with open(os.path.join(final_save_path, 'final_training_info.json'), 'w') as f:
            json.dump(final_training_info, f, indent=2)
        
        print(f"✅ Final model saved to {final_save_path}")
        print_memory_usage(rank, "final save")
    
    # Cleanup
    cleanup_distributed()

def main():
    """Main function to launch FORCED 2 GPU distributed training."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Distributed training requires GPUs.")
        return
    
    # Check if we have at least 2 GPUs
    available_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {available_gpus} GPUs")
    
    if available_gpus < 2:
        print(f"❌ Need at least 2 GPUs for distributed training. Found {available_gpus} GPUs.")
        print("💡 This script is designed to FORCE 2 GPU distributed training.")
        return
    
    # Force 2 GPU training
    actual_world_size = 2
    print(f"🚀 FORCING distributed training on {actual_world_size} GPUs (Pre-trained LoRA Model)")
    print(f"📊 Dataset: {DATASET_NAME}")
    print(f"📊 Model: {MODEL_NAME}")
    print(f"📊 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Effective batch size: {BATCH_SIZE * actual_world_size * GRADIENT_ACCUMULATION_STEPS}")
    print(f"📊 Note: Using pre-trained LoRA model (no quantization needed)")
    
    # Check GPU memory
    for i in range(actual_world_size):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"🔍 GPU {i}: {gpu_memory:.1f}GB total memory")
        if gpu_memory < 14:  # Leave some headroom
            print(f"⚠️ GPU {i} has less than 14GB memory. Training might fail.")
    
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
        print("❌ Cannot run distributed training in interactive environment")
        print("💡 Please run: python distributed_trainer_hub_force2gpu.py")

if __name__ == "__main__":
    main()

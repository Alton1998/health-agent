import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
import os
import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def train_worker(rank, world_size, args):
    """
    Main training function for each distributed worker.
    
    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42 + rank)  # Different seed for each process
    
    # Device setup
    device = torch.device(f'cuda:{rank}')
    
    # Only rank 0 should print setup information
    if rank == 0:
        print(f"🔧 Training Configuration:")
        print(f"   - Model: {args.model_name}")
        print(f"   - World Size: {world_size}")
        print(f"   - Batch Size: {args.batch_size}")
        print(f"   - Learning Rate: {args.learning_rate}")
        print(f"   - Epochs: {args.num_epochs}")
        print(f"   - Gradient Accumulation: {args.gradient_accumulation_steps}")
    
    # --- Model & Tokenizer Setup ---
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048
    
    # --- Dataset Loading ---
    if rank == 0:
        print("📊 Loading dataset...")
    
    dataset = load_from_disk(args.dataset_path)
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
        print("🤖 Loading model...")
    
    # Bits and Bytes configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=None,  # We'll move to device manually
        trust_remote_code=True,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    peft_model = get_peft_model(model, peft_config)
    
    if rank == 0:
        peft_model.print_trainable_parameters()
    
    # Move model to device
    peft_model = peft_model.to(device)
    
    # Wrap model with DDP
    peft_model = DDP(peft_model, device_ids=[rank], output_device=rank)
    
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
        test_dataset.select(range(500)), 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=2,
        pin_memory=True
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        test_dataset.select(range(500)),
        batch_size=args.batch_size,
        sampler=eval_sampler,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
        num_workers=2,
        pin_memory=True
    )
    
    # --- Training Setup ---
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training state
    global_step = 0
    best_eval_loss = float('inf')
    
    # --- Training Loop ---
    if rank == 0:
        print("🚀 Starting training...")
    
    for epoch in range(args.num_epochs):
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
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            if torch.isnan(loss):
                if rank == 0:
                    print(f"⚠️ Skipping NaN loss at step {step}")
                continue
            
            # Scale loss for gradient accumulation
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            
            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), args.max_grad_norm)
                
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
            if rank == 0 and global_step % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                progress.set_postfix(loss=avg_loss)
                running_loss = 0.0
            
            # --- Evaluation ---
            if global_step % args.eval_steps == 0 and global_step > 0:
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
                    # Gather evaluation losses from all processes
                    eval_losses_tensor = torch.tensor(eval_losses, device=device)
                    gathered_losses = [torch.zeros_like(eval_losses_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered_losses, eval_losses_tensor)
                    
                    # Calculate mean across all processes
                    all_losses = torch.cat(gathered_losses).cpu().numpy()
                    mean_eval_loss = all_losses.mean()
                    
                    if rank == 0:
                        print(f"✅ Eval at step {global_step} | loss: {mean_eval_loss:.4f}")
                        
                        # Save best model if eval loss improved
                        if mean_eval_loss < best_eval_loss:
                            best_eval_loss = mean_eval_loss
                            save_path = f"{args.output_dir}/checkpoint-best"
                            
                            # Save model (only rank 0)
                            peft_model.module.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            
                            print(f"💾 New best model saved at step {global_step} with loss: {best_eval_loss:.4f}")
                
                peft_model.train()
            
            # --- Save Checkpoint ---
            if global_step % args.save_steps == 0 and global_step > 0 and rank == 0:
                save_path = f"{args.output_dir}/checkpoint-{global_step}"
                peft_model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"💾 Checkpoint saved at step {global_step}")
        
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
    
    # --- Final Save ---
    if rank == 0:
        print("💾 Saving final model...")
        final_save_path = args.output_dir
        peft_model.module.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"✅ Final model saved to {final_save_path}")
    
    # Cleanup
    cleanup_distributed()

def main():
    """Main function to launch distributed training."""
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    
    # Model and data arguments
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Model name or path')
    parser.add_argument('--dataset_path', type=str, default='./health-agent-dataset',
                       help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./health-agent-distributed',
                       help='Output directory for checkpoints')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=0.3,
                       help='Maximum gradient norm for clipping')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Training control arguments
    parser.add_argument('--eval_steps', type=int, default=50,
                       help='Evaluation frequency in steps')
    parser.add_argument('--save_steps', type=int, default=50,
                       help='Save frequency in steps')
    parser.add_argument('--log_steps', type=int, default=1,
                       help='Logging frequency in steps')
    
    # Distributed training arguments
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Distributed training requires GPUs.")
        return
    
    # Check if we have enough GPUs
    available_gpus = torch.cuda.device_count()
    if args.world_size > available_gpus:
        print(f"⚠️ Requested {args.world_size} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        args.world_size = available_gpus
    
    print(f"🚀 Starting distributed training on {args.world_size} GPUs")
    print(f"📊 Available GPUs: {available_gpus}")
    
    # Launch distributed training
    mp.spawn(
        train_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()

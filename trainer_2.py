import logging
import torch
import math
import os
import json
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

torch.manual_seed(42)
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# --- Model & Tokenizer ---
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

# --- Dataset ---
dataset = load_from_disk("./health-agent-dataset")
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Shuffle the datasets
train_dataset = train_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# --- Tokenization ---
def tokenize_function(example):
    try:
        messages = example["messages"]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,  # Don't pad here, let the collator handle it
            return_tensors=None  # Return lists, not tensors
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.copy()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    except Exception as e:
        print(f"Error tokenizing example: {e}")
        # Return a minimal valid example as lists
        return {
            "input_ids": [tokenizer.eos_token_id],
            "attention_mask": [1],
            "labels": [tokenizer.eos_token_id]
        }

# Remove all columns before tokenization to ensure clean data
train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

# Verify the data structure
print("Train dataset columns after tokenization:", train_dataset.column_names)
print("Test dataset columns after tokenization:", test_dataset.column_names)
print("Sample train data keys:", train_dataset[0].keys() if len(train_dataset) > 0 else "No data")

# Verify tensor types
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print("Sample tensor types:")
    print(f"input_ids type: {type(sample['input_ids'])}")
    print(f"attention_mask type: {type(sample['attention_mask'])}")
    print(f"labels type: {type(sample['labels'])}")
    if hasattr(sample['input_ids'], 'dtype'):
        print(f"input_ids dtype: {sample['input_ids'].dtype}")
        print(f"attention_mask dtype: {sample['attention_mask'].dtype}")
        print(f"labels dtype: {sample['labels'].dtype}")

# --- Custom Data Collator ---
def custom_collate_fn(batch):
    # Find the maximum length in this batch
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Pad all sequences to the same length
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for i, item in enumerate(batch):
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]
        
        # Debug: Check data types
        if i == 0:  # Only print for first item to avoid spam
            print(f"🔍 Batch item {i} types: input_ids={type(input_ids)}, attention_mask={type(attention_mask)}, labels={type(labels)}")
        
        # Convert to tensors if they're lists
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Ensure tensors are on CPU and don't require gradients
        input_ids = input_ids.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        labels = labels.detach().cpu()
        
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

# --- Bits and Bytes + LoRA ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device,  # Use single device instead of auto
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# Check for trainer checkpoint and load if available
trainer_checkpoint_path = "./health-agent-finetuned/trainer_checkpoint"
best_checkpoint_path = "./health-agent-finetuned/checkpoint-best"

# Training configuration
num_epochs = 5
gradient_accumulation_steps = 4
eval_steps = 50
log_steps = 1
save_steps = 50  # Save every 50 steps
max_grad_norm = 0.3

# Early stopping configuration
early_stopping_patience = 3  # Number of epochs to wait before stopping
early_stopping_counter = 0
best_epoch_loss = float('inf')

# Initialize training variables
global_step = 0
best_eval_loss = float('inf')
start_epoch = 0
start_batch = 0  # Track which batch to start from within the epoch

# Try to load trainer checkpoint first (most complete)
if os.path.exists(trainer_checkpoint_path):
    print(f"🔄 Loading trainer checkpoint from {trainer_checkpoint_path}")
    try:
        # Load the adapter weights
        peft_model.load_adapter(trainer_checkpoint_path, adapter_name="default")
        
        # Load complete training state
        trainer_state_path = os.path.join(trainer_checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            import json
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                global_step = trainer_state.get('global_step', 0)
                best_eval_loss = trainer_state.get('best_eval_loss', float('inf'))
                if best_eval_loss == 'inf':
                    best_eval_loss = float('inf')
                start_epoch = trainer_state.get('epoch', 0)
                best_epoch_loss = trainer_state.get('best_epoch_loss', float('inf'))
                if best_epoch_loss == 'inf':
                    best_epoch_loss = float('inf')
                early_stopping_counter = trainer_state.get('early_stopping_counter', 0)
                
                # Calculate which batch to start from within the epoch
                batches_per_epoch = len(train_dataset) // 1  # batch_size = 1
                completed_batches_in_epoch = global_step * gradient_accumulation_steps % batches_per_epoch
                start_batch = completed_batches_in_epoch
                print(f"📊 Resuming from epoch {start_epoch}, step {global_step}")
                print(f"📊 Best eval loss so far: {best_eval_loss}")
                print(f"📊 Best epoch loss so far: {best_epoch_loss}")
                print(f"📊 Early stopping counter: {early_stopping_counter}")
        
        print("✅ Successfully loaded trainer checkpoint")
        
    except Exception as e:
        print(f"⚠️ Error loading trainer checkpoint: {e}")
        print("🔄 Trying to load best model checkpoint...")
        
        # Fallback to best checkpoint
        if os.path.exists(best_checkpoint_path):
            try:
                peft_model.load_adapter(best_checkpoint_path, adapter_name="default")
                print("✅ Successfully loaded best model checkpoint")
            except Exception as e2:
                print(f"⚠️ Error loading best checkpoint: {e2}")
                print("🔄 Starting training from scratch")
                global_step = 0
                best_eval_loss = float('inf')
                start_epoch = 0
        else:
            print("🆕 No checkpoints found, starting training from scratch")
            global_step = 0
            best_eval_loss = float('inf')
            start_epoch = 0

else:
    print("🆕 No trainer checkpoint found, starting training from scratch")

# Ensure model is on the correct device and trainable
peft_model.train()
print(f"Model device: {next(peft_model.parameters()).device}")
print(f"Target device: {device}")

# Verify target modules are trainable
print("🔍 Checking LoRA target modules:")
for name, param in peft_model.named_parameters():
    # Only check LoRA adapter parameters, not base layer parameters
    if any(target in name for target in ["q_proj", "v_proj", "gate_proj", "up_proj"]) and "lora" in name.lower():
        if param.requires_grad:
            print(f"✅ {name} - Trainable")
        else:
            print(f"❌ {name} - Not trainable")
            if param.dtype in [torch.float16, torch.float32, torch.float64]:
                param.requires_grad_(True)
                print(f"🔧 Enabled gradients for {name}")
            else:
                print(f"⚠️ Cannot enable gradients for {name} (dtype: {param.dtype})")

# --- DataLoaders ---
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,  # Reduced from 64 to 8 for memory efficiency
    shuffle=True,
    collate_fn=custom_collate_fn,
)

eval_dataloader = DataLoader(
    test_dataset.select(range(500)),
    batch_size=1,  # Reduced from 32 to 4 for evaluation
    collate_fn=custom_collate_fn,
)

# --- Training Loop ---
from torch.cuda.amp import autocast, GradScaler

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=5e-5)
scaler = GradScaler()

for epoch in range(start_epoch, num_epochs):
    peft_model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    epoch_loss = 0.0
    num_batches = 0

    # Skip batches that were already completed in this epoch
    if epoch == start_epoch and start_batch > 0:
        print(f"🔄 Skipping first {start_batch} batches in epoch {epoch}")
        # Create a custom iterator that skips the first start_batch items
        dataloader_iter = iter(train_dataloader)
        for _ in range(start_batch):
            next(dataloader_iter)
        # Use enumerate to get both step and batch
        progress = tqdm(enumerate(dataloader_iter, start=start_batch), total=len(train_dataloader) - start_batch, desc=f"Epoch {epoch}")
    else:
        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
    
    for step, batch in progress:
        # Ensure all tensors are on the same device as the model
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast():
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if torch.isnan(loss):
            print(f"⚠️ Skipping NaN loss at step {step}")
            continue

        scaler.scale(loss / gradient_accumulation_steps).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_grad_norm)

            if not torch.isnan(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                print(f"⚠️ Skipping optimizer step due to NaN grad_norm at step {step}")

            global_step += 1

        # Accumulate losses
        running_loss += loss.item()  # For progress bar updates
        epoch_loss += loss.item()    # For epoch-level early stopping
        num_batches += 1

        if global_step % log_steps == 0:
            avg_loss = running_loss / log_steps
            progress.set_postfix(loss=avg_loss)
            running_loss = 0.0  # Reset running loss for next log interval

        # --- Save Checkpoint ---
        if global_step % save_steps == 0 and global_step > 0:
            # Save regular checkpoint
            save_path = f"./health-agent-finetuned/checkpoint-{global_step}"
            peft_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save training state (convert tensors to Python types)
            trainer_state = {
                'global_step': global_step,
                'best_eval_loss': float(best_eval_loss) if best_eval_loss != float('inf') else 'inf',
                'epoch': epoch,
                'best_epoch_loss': float(best_epoch_loss) if best_epoch_loss != float('inf') else 'inf',
                'early_stopping_counter': early_stopping_counter
            }
            with open(os.path.join(save_path, 'trainer_state.json'), 'w') as f:
                json.dump(trainer_state, f, indent=2)
            
            print(f"💾 Checkpoint saved at step {global_step}")
            
            # Save trainer checkpoint (for resuming)
            trainer_save_path = "./health-agent-finetuned/trainer_checkpoint"
            peft_model.save_pretrained(trainer_save_path)
            tokenizer.save_pretrained(trainer_save_path)
            
            # Save complete trainer state (convert tensors to Python types)
            trainer_state = {
                'global_step': global_step,
                'best_eval_loss': float(best_eval_loss) if best_eval_loss != float('inf') else 'inf',
                'epoch': epoch,
                'best_epoch_loss': float(best_epoch_loss) if best_epoch_loss != float('inf') else 'inf',
                'early_stopping_counter': early_stopping_counter
            }
            with open(os.path.join(trainer_save_path, 'trainer_state.json'), 'w') as f:
                json.dump(trainer_state, f, indent=2)
            
            print(f"💾 Trainer checkpoint saved at step {global_step}")

        # --- Evaluation ---
        if global_step % eval_steps == 0:
            peft_model.eval()
            eval_losses = []

            with torch.no_grad():
                for eval_batch in eval_dataloader:
                    ids = eval_batch["input_ids"].to(device, non_blocking=True)
                    mask = eval_batch["attention_mask"].to(device, non_blocking=True)
                    labels = eval_batch["labels"].to(device, non_blocking=True)

                    with autocast():
                        outputs = peft_model(input_ids=ids, attention_mask=mask, labels=labels)
                        eval_loss = outputs.loss

                    if not torch.isnan(eval_loss):
                        eval_losses.append(eval_loss.item())

            if eval_losses:
                mean_eval_loss = sum(eval_losses) / len(eval_losses)
                print(f"✅ Eval at step {global_step} | loss: {mean_eval_loss:.4f}")
                
                # Save best model if eval loss improved
                if mean_eval_loss < best_eval_loss:
                    best_eval_loss = mean_eval_loss
                    save_path = f"./health-agent-finetuned/checkpoint-best"
                    peft_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"💾 New best model saved at step {global_step} with loss: {best_eval_loss:.4f}")
            else:
                print(f"⚠️ Skipping eval at step {global_step} due to NaN losses")

            peft_model.train()

    # Calculate average loss for this epoch
    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
    print(f"📊 Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
    
    # Early stopping check
    if avg_epoch_loss < best_epoch_loss:
        best_epoch_loss = avg_epoch_loss
        early_stopping_counter = 0
        print(f"🎯 New best epoch loss: {best_epoch_loss:.4f}")
    else:
        early_stopping_counter += 1
        print(f"⏳ No improvement for {early_stopping_counter} epochs. Best: {best_epoch_loss:.4f}")
    
    # Check if we should stop early
    if early_stopping_counter >= early_stopping_patience:
        print(f"🛑 Early stopping triggered after {early_stopping_patience} epochs without improvement")
        break

# --- ✅ Save Model & Tokenizer ---
save_path = "./health-agent-finetuned"
peft_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Model and tokenizer saved to {save_path}")

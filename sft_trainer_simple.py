import logging
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

# --- Model & Tokenizer ---
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

# --- Dataset ---
dataset = load_from_disk("./health-agent-dataset")
split_dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# --- Preprocess data for SFTTrainer ---
def format_for_sft(example):
    """Convert messages to the format expected by SFTTrainer"""
    messages = example["messages"]
    # Apply chat template to get the formatted text
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_text}

# Preprocess datasets
train_dataset = train_dataset.map(format_for_sft, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(format_for_sft, remove_columns=test_dataset.column_names)

print("After preprocessing:")
print(f"Train dataset columns: {train_dataset.column_names}")
print(f"Test dataset columns: {test_dataset.column_names}")

# Print sample
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print("Sample text (first 200 chars):")
    print(sample["text"][:200] + "...")

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
    device_map=device,  # Use single device
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

# Enable gradient computation for training
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

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

# Ensure the model is in training mode
peft_model.train()

# Ensure model is on the correct device
print(f"Model device: {next(peft_model.parameters()).device}")
print(f"Target device: {device}")

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./health-agent-sft-simple-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=5,
    save_steps=5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_pin_memory=False,
    max_grad_norm=0.3,
    optim="paged_adamw_32bit",
    remove_unused_columns=False,
)

# --- SFT Trainer ---
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Starting SFT Training (Simple Version)")
trainer.train()

# Save the model
trainer.save_model()
tokenizer.save_pretrained("./health-agent-sft-simple-finetuned")
print("✅ Model and tokenizer saved to ./health-agent-sft-simple-finetuned") 
import torch.cuda
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
    EarlyStoppingCallback
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and base model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Named modules in the model:")
for name, module in model.named_modules():
    print(name)

# Load datasets
train_dataset = load_dataset("json", data_files="tokenized_train_data.jsonl", split="train")
test_dataset = load_dataset("json", data_files="tokenized_test_data.jsonl", split="train")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Tokenize the data
def tokenize_function(example):
    # Clean the text by removing problematic Unicode characters
    text = example["text"]
            # .replace("\uff5c", "|").replace("\u2581", "_"))
    
    # Tokenize
    result = tokenizer(
        text,
        truncation=True,
        max_length=3600,
        padding="max_length",
        return_tensors=None
    )
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = train_dataset.map(tokenize_function, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_function, remove_columns=["text"])

# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA config for QLoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        # "q_proj",
        # "k_proj",
        # "v_proj",
        # "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap model with PEFT
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./health-agent-finetuned',
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
    save_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_pin_memory=False,
)

# Regular Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

print("Starting Training")
trainer.train()

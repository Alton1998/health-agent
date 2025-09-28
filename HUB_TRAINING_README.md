# Health Agent Training with Hub Dataset

This guide explains how to use the new `trainer_hub_dataset.py` script to train your health agent model using the dataset from Hugging Face Hub.

## 🚀 Quick Start

### **Run Training:**
```bash
# Option A: Direct Python script
python trainer_hub_dataset.py

# Option B: Use launcher (Linux/Mac)
chmod +x train_hub_dataset.sh
./train_hub_dataset.sh

# Option C: Use launcher (Windows)
train_hub_dataset.bat
```

## 📊 What's Different

### **Key Changes from `trainer_2.py`:**

| Aspect | Original `trainer_2.py` | New `trainer_hub_dataset.py` |
|--------|------------------------|------------------------------|
| **Dataset Source** | Local disk (`./health-agent-dataset`) | Hugging Face Hub (`aldsouza/health-agent-dataset`) |
| **Loading Method** | `load_from_disk()` | `load_dataset()` |
| **Output Directory** | `./health-agent-finetuned` | `./health-agent-hub-trained` |
| **Device** | `cuda:1` | `cuda:0` |
| **Training Info** | Basic checkpoint info | Includes dataset and model names |

## 🔧 Configuration

### **Dataset Configuration:**
```python
DATASET_NAME = "aldsouza/health-agent-dataset"  # Your Hub dataset
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "./health-agent-hub-trained"
```

### **Training Parameters:**
```python
num_epochs = 5
gradient_accumulation_steps = 4
eval_steps = 50
save_steps = 50
learning_rate = 5e-5
batch_size = 1
```

## 📋 Dataset Information

### **Source Dataset:**
- **Repository**: `aldsouza/health-agent-dataset`
- **Size**: ~60,000 samples
- **Format**: Chat messages with function calling
- **Content**: Health agent training data

### **Dataset Structure:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an intelligent AI assistant..."
    },
    {
      "role": "user",
      "content": "User query here"
    },
    {
      "role": "assistant",
      "content": "[{\"name\": \"function_name\", \"arguments\": {...}}]"
    }
  ]
}
```

## 🎯 Training Process

### **Step-by-Step:**
1. **Load Dataset**: Downloads from Hugging Face Hub
2. **Tokenize**: Processes chat messages into tokens
3. **Setup Model**: Loads model with 4-bit quantization and LoRA
4. **Train**: Runs training loop with checkpointing
5. **Evaluate**: Regular evaluation during training
6. **Save**: Saves best model and final checkpoint

### **Training Output:**
```
🤖 Loading model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
📊 Loading dataset from Hub: aldsouza/health-agent-dataset
✅ Successfully loaded dataset from Hub
📊 Train dataset size: 48000
📊 Test dataset size: 12000
🚀 Starting training with dataset from Hub: aldsouza/health-agent-dataset
```

## 💾 Checkpoint Management

### **Checkpoint Types:**
- **Regular Checkpoints**: `./health-agent-hub-trained/checkpoint-{step}`
- **Best Model**: `./health-agent-hub-trained/checkpoint-best`
- **Trainer Checkpoint**: `./health-agent-hub-trained/trainer_checkpoint`
- **Final Model**: `./health-agent-hub-trained/`

### **Checkpoint Contents:**
```json
{
  "global_step": 150,
  "best_eval_loss": 0.234,
  "epoch": 2,
  "best_epoch_loss": 0.456,
  "early_stopping_counter": 0,
  "dataset_name": "aldsouza/health-agent-dataset",
  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
}
```

## 🔄 Resuming Training

### **Automatic Resume:**
The script automatically resumes from the last checkpoint if available:
```
🔄 Loading trainer checkpoint from ./health-agent-hub-trained/trainer_checkpoint
📊 Resuming from epoch 2, step 150
📊 Best eval loss so far: 0.234
```

### **Manual Resume:**
If you want to start fresh, delete the checkpoint directory:
```bash
rm -rf ./health-agent-hub-trained
```

## 📊 Monitoring Training

### **Progress Indicators:**
- **Epoch Progress**: Shows current epoch and batch progress
- **Loss Tracking**: Running loss and average loss per epoch
- **Evaluation**: Regular evaluation with loss reporting
- **Checkpoint Saving**: Automatic saving at specified intervals

### **Example Output:**
```
Epoch 1: 100%|██████████| 12000/12000 [15:30<00:00, 12.9it/s, loss=0.456]
✅ Eval at step 50 | loss: 0.234
💾 New best model saved at step 50 with loss: 0.234
📊 Epoch 1 average loss: 0.456
```

## 🛠️ Troubleshooting

### **Common Issues:**

1. **Dataset Not Found:**
   ```
   ❌ Error loading dataset from Hub: Repository not found
   ```
   **Solution**: Check if `aldsouza/health-agent-dataset` exists and is accessible

2. **CUDA Out of Memory:**
   ```
   ❌ CUDA out of memory
   ```
   **Solution**: Reduce batch size or use gradient checkpointing

3. **Authentication Error:**
   ```
   ❌ Error: Not authenticated
   ```
   **Solution**: Login to Hugging Face Hub or use public dataset

4. **Slow Dataset Loading:**
   ```
   ⏳ Loading dataset from Hub...
   ```
   **Solution**: First load may be slow, subsequent loads will be cached

### **Debug Mode:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

### **Memory Optimization:**
- Uses 4-bit quantization (QLoRA)
- Batch size of 1 with gradient accumulation
- Automatic mixed precision training
- Gradient checkpointing enabled

### **Training Speed:**
- Uses CUDA if available
- Optimized data loading
- Efficient tokenization
- Smart checkpointing

## 🔗 Integration with Other Scripts

### **Use with Distributed Training:**
```python
# Update distributed_trainer.py to use Hub dataset
dataset = load_dataset("aldsouza/health-agent-dataset")
```

### **Use with Evaluation:**
```python
# Load trained model
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = PeftModel.from_pretrained(model, "./health-agent-hub-trained")
```

## 📚 File Structure

```
health-agent/
├── trainer_hub_dataset.py     # Main training script
├── train_hub_dataset.sh       # Linux/Mac launcher
├── train_hub_dataset.bat      # Windows launcher
├── health-agent-hub-trained/  # Training output directory
│   ├── checkpoint-best/       # Best model checkpoint
│   ├── checkpoint-50/         # Regular checkpoints
│   ├── trainer_checkpoint/    # Resume checkpoint
│   └── training_info.json     # Final training info
└── HUB_TRAINING_README.md     # This documentation
```

## 🎉 Benefits of Hub Dataset

1. **Always Up-to-Date**: Uses latest version from Hub
2. **No Local Storage**: No need to store dataset locally
3. **Easy Sharing**: Others can use the same dataset
4. **Version Control**: Track dataset changes
5. **Cloud Access**: Access from anywhere

## 🔄 Workflow Integration

### **Complete Workflow:**
1. **Create Dataset**: Upload to Hub using `push_dataset_to_hub.py`
2. **Train Model**: Use `trainer_hub_dataset.py`
3. **Evaluate**: Test the trained model
4. **Deploy**: Use in production

### **Update Other Scripts:**
```python
# Old way (local dataset)
dataset = load_from_disk("./health-agent-dataset")

# New way (Hub dataset)
dataset = load_dataset("aldsouza/health-agent-dataset")
```

## 📖 Usage Examples

### **Basic Training:**
```bash
python trainer_hub_dataset.py
```

### **With Custom Configuration:**
```python
# Modify the script to change:
DATASET_NAME = "your-username/your-dataset"
MODEL_NAME = "your-preferred-model"
OUTPUT_DIR = "./your-output-directory"
```

### **Load Trained Model:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./health-agent-hub-trained")
```

This new training script provides a seamless way to train your health agent model using the dataset from Hugging Face Hub, with all the benefits of cloud-based dataset management!

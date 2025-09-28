# Simple Distributed Training with Hardcoded Arguments

This guide explains how to use the simplified `distributed_trainer_hub_simple.py` script that has all arguments hardcoded for easy execution.

## 🚀 Quick Start

### **Run Training (No Arguments Needed):**
```bash
# Option A: Direct Python script
python distributed_trainer_hub_simple.py

# Option B: Use launcher (Linux/Mac)
chmod +x launch_distributed_simple.sh
./launch_distributed_simple.sh

# Option C: Use launcher (Windows)
launch_distributed_simple.bat
```

## 🔧 Hardcoded Configuration

### **All Settings Pre-Configured:**
```python
# Model and Dataset
MODEL_NAME = "aldsouza/health-agent-lora-rkllm-compatible"
DATASET_NAME = "aldsouza/health-agent-dataset"
OUTPUT_DIR = "./health-agent-distributed-hub"

# Training (Optimized for 2x 16GB GPUs)
NUM_EPOCHS = 5
BATCH_SIZE = 1  # Per GPU
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 0.3
MAX_LENGTH = 1024

# LoRA (Model already has LoRA adapters)
# LORA_R = 8
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.1

# Training Control
EVAL_STEPS = 50
SAVE_STEPS = 50
LOG_STEPS = 1

# Distributed
WORLD_SIZE = 2  # Number of GPUs
```

## 📊 What This Script Does

### **Automatic Configuration:**
- ✅ **Detects Available GPUs**: Automatically uses all available GPUs
- ✅ **Loads Hub Dataset**: Downloads from `aldsouza/health-agent-dataset`
- ✅ **Pre-trained LoRA Model**: Uses `aldsouza/health-agent-lora-rkllm-compatible`
- ✅ **No Quantization**: Model is already optimized
- ✅ **No Arguments**: Just run the script!

### **Training Process:**
1. **Auto-Detect GPUs**: Uses all available GPUs (up to 2)
2. **Load Dataset**: Downloads from Hugging Face Hub
3. **Setup Model**: Loads pre-trained LoRA model (no quantization needed)
4. **Distributed Training**: Runs across multiple GPUs
5. **Save Checkpoints**: Automatic saving and best model tracking

## 🎯 Key Benefits

### **1. Zero Configuration:**
- No command line arguments needed
- All settings optimized for your setup
- Just run and train!

### **2. Smart GPU Detection:**
```python
# Automatically detects and uses available GPUs
available_gpus = torch.cuda.device_count()
if WORLD_SIZE > available_gpus:
    actual_world_size = available_gpus  # Use what's available
```

### **3. Memory Optimized:**
- Pre-trained LoRA model (already optimized)
- Gradient checkpointing enabled
- Mixed precision training
- Optimized batch sizes

### **4. Hub Dataset Integration:**
- Loads from `aldsouza/health-agent-dataset`
- No local dataset storage needed
- Always uses latest version

## 📋 Training Output

### **Expected Output:**
```
🚀 Health Agent Distributed Training (Simple Version)
=====================================================
🔍 Available GPUs: 2
✅ Dependencies found
🚀 Starting distributed training with hardcoded configuration...
📊 Configuration:
   - Model: aldsouza/health-agent-lora-rkllm-compatible
   - Dataset: aldsouza/health-agent-dataset
   - GPUs: 2
   - Batch size per GPU: 1
   - Gradient accumulation: 8
   - Learning rate: 5e-5
   - Epochs: 5
   - Note: Using pre-trained LoRA model (no quantization)

🔧 Distributed Training Configuration (Pre-trained LoRA Model):
   - Model: aldsouza/health-agent-lora-rkllm-compatible
   - Dataset: aldsouza/health-agent-dataset
   - World Size: 2
   - Batch Size per GPU: 1
   - Effective Batch Size: 16
   - Learning Rate: 5e-5
   - Epochs: 5
   - Gradient Accumulation: 8
   - Max Sequence Length: 1024

📊 Loading dataset from Hub: aldsouza/health-agent-dataset
✅ Successfully loaded dataset from Hub
   - Train dataset size: 48000
   - Test dataset size: 12000
🤖 Loading pre-trained LoRA model...
📊 Trainable parameters: 4,194,304
📊 Total parameters: 1,500,000,000
📊 Trainable %: 0.28%
🚀 Starting distributed training...
```

## 💾 Output Files

### **Training Output:**
```
health-agent-distributed-hub/
├── checkpoint-best/              # Best model checkpoint
├── checkpoint-50/                # Regular checkpoints
├── checkpoint-100/
├── training_info.json            # Training information
└── final_training_info.json      # Final training info
```

### **Training Info:**
```json
{
  "model_name": "aldsouza/health-agent-lora-rkllm-compatible",
  "dataset_name": "aldsouza/health-agent-dataset",
  "global_step": 150,
  "best_eval_loss": 0.234,
  "world_size": 2,
  "batch_size": 1,
  "learning_rate": 5e-5,
  "training_completed": true
}
```

## 🔧 Customization

### **To Modify Settings:**
Edit the configuration section at the top of `distributed_trainer_hub_simple.py`:

```python
# =============================================================================
# HARDCODED CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# =============================================================================

# Model and Dataset Configuration
MODEL_NAME = "your-model-name"           # Change model
DATASET_NAME = "your-dataset-name"       # Change dataset
OUTPUT_DIR = "./your-output-directory"   # Change output directory

# Training Configuration
NUM_EPOCHS = 10                          # Change epochs
BATCH_SIZE = 2                           # Change batch size
LEARNING_RATE = 1e-4                     # Change learning rate

# LoRA Configuration (Model already has LoRA adapters)
# LORA_R = 16                              # Change LoRA rank
# LORA_ALPHA = 32                          # Change LoRA alpha

# Distributed Training
WORLD_SIZE = 4                           # Change number of GPUs
```

## 🛠️ Troubleshooting

### **Common Issues:**

1. **CUDA Not Available:**
   ```
   ❌ CUDA is not available. Distributed training requires GPUs.
   ```
   **Solution**: Install PyTorch with CUDA support

2. **Dataset Not Found:**
   ```
   ❌ Error loading dataset from Hub: Repository not found
   ```
   **Solution**: Check if `aldsouza/health-agent-dataset` exists

3. **Out of Memory:**
   ```
   ❌ CUDA out of memory
   ```
   **Solution**: Reduce `BATCH_SIZE` or increase `GRADIENT_ACCUMULATION_STEPS`

4. **Not Enough GPUs:**
   ```
   ⚠️ Requested 2 GPUs but only 1 available. Using 1 GPUs.
   ```
   **Solution**: Script automatically adjusts to available GPUs

## 📈 Performance

### **Expected Performance:**
- **2x 16GB GPUs**: ~1.8x speedup
- **Memory Usage**: ~6-8GB per GPU (50-60% utilization)
- **Effective Batch Size**: 16 (1 × 2 GPUs × 8 accumulation)
- **Training Time**: ~45% reduction vs single GPU

### **Memory Breakdown:**
```
GPU 0: ~6-8GB used / 16GB total
GPU 1: ~6-8GB used / 16GB total
```

## 🔄 Workflow Integration

### **Complete Workflow:**
1. **Create Dataset**: Upload to Hub using `push_dataset_to_hub.py`
2. **Simple Training**: Run `distributed_trainer_hub_simple.py`
3. **Evaluate**: Test the trained model
4. **Deploy**: Use in production

### **Use Trained Model:**
```python
from transformers import AutoModelForCausalLM

# Load the trained model directly (already has LoRA adapters)
model = AutoModelForCausalLM.from_pretrained("./health-agent-distributed-hub")
```

## 📚 File Structure

```
health-agent/
├── distributed_trainer_hub_simple.py    # Main script (hardcoded)
├── launch_distributed_simple.sh         # Linux/Mac launcher
├── launch_distributed_simple.bat        # Windows launcher
├── health-agent-distributed-hub/        # Training output
└── SIMPLE_DISTRIBUTED_TRAINING_README.md # This documentation
```

## 🎉 Benefits of Hardcoded Version

1. **Zero Configuration**: No arguments needed
2. **Easy to Use**: Just run the script
3. **Optimized Settings**: Pre-configured for best performance
4. **Auto-Detection**: Automatically uses available GPUs
5. **Error Handling**: Graceful handling of common issues
6. **Production Ready**: Robust and reliable

## 📖 Usage Examples

### **Basic Usage:**
```bash
python distributed_trainer_hub_simple.py
```

### **With Launcher:**
```bash
# Linux/Mac
./launch_distributed_simple.sh

# Windows
launch_distributed_simple.bat
```

### **Custom Configuration:**
Edit the configuration section in the script and run:
```bash
python distributed_trainer_hub_simple.py
```

This simplified version makes distributed training as easy as running a single command! Perfect for users who want to get started quickly without worrying about configuration details.

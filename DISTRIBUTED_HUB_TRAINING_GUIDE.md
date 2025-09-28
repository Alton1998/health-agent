# Distributed Training with Hub Dataset Guide

This guide explains how to use the new `distributed_trainer_hub.py` script for distributed training using the dataset from Hugging Face Hub.

## 🚀 Quick Start

### **Launch Distributed Training:**
```bash
# Option A: Use launcher script (Linux/Mac)
chmod +x launch_distributed_hub.sh
./launch_distributed_hub.sh 2

# Option B: Use launcher script (Windows)
launch_distributed_hub.bat 2

# Option C: Direct Python script
python distributed_trainer_hub.py --world_size 2 --dataset_name aldsouza/health-agent-dataset
```

## 📊 What's Different

### **Key Changes from `distributed_trainer.py`:**

| Aspect | Original `distributed_trainer.py` | New `distributed_trainer_hub.py` |
|--------|-----------------------------------|----------------------------------|
| **Dataset Source** | Local disk (`./health-agent-dataset`) | Hugging Face Hub (`aldsouza/health-agent-dataset`) |
| **Loading Method** | `load_from_disk()` | `load_dataset()` |
| **Output Directory** | `./health-agent-distributed` | `./health-agent-distributed-hub` |
| **Training Info** | Basic checkpoint info | Includes dataset and model names |
| **Error Handling** | Basic dataset loading | Enhanced Hub dataset error handling |

## 🔧 Configuration

### **Default Configuration:**
```python
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "aldsouza/health-agent-dataset"
OUTPUT_DIR = "./health-agent-distributed-hub"
WORLD_SIZE = 2  # Number of GPUs
BATCH_SIZE = 1  # Per GPU
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
```

### **Command Line Arguments:**
```bash
python distributed_trainer_hub.py \
    --world_size 2 \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset_name "aldsouza/health-agent-dataset" \
    --output_dir "./health-agent-distributed-hub" \
    --num_epochs 5 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 8 \
    --max_length 1024
```

## 📋 Dataset Information

### **Hub Dataset:**
- **Repository**: `aldsouza/health-agent-dataset`
- **Size**: ~60,000 samples
- **Format**: Chat messages with function calling
- **Access**: Public dataset on Hugging Face Hub

### **Dataset Loading Process:**
```python
# Load dataset from Hub
dataset = load_dataset("aldsouza/health-agent-dataset")

# Split and shuffle
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"].shuffle(seed=42)
test_dataset = split_dataset["test"].shuffle(seed=42)
```

## 🎯 Training Process

### **Distributed Training Flow:**
1. **Initialize**: Setup distributed environment across GPUs
2. **Load Dataset**: Download from Hugging Face Hub
3. **Tokenize**: Process chat messages into tokens
4. **Setup Model**: Load with 4-bit quantization and LoRA
5. **Distribute**: Wrap model with DistributedDataParallel
6. **Train**: Run distributed training loop
7. **Evaluate**: Gather evaluation metrics across GPUs
8. **Save**: Save checkpoints (rank 0 only)

### **Training Output:**
```
🔧 Distributed Training Configuration (Hub Dataset):
   - Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
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
🚀 Starting distributed training...
```

## 💾 Checkpoint Management

### **Checkpoint Types:**
- **Regular Checkpoints**: `./health-agent-distributed-hub/checkpoint-{step}`
- **Best Model**: `./health-agent-distributed-hub/checkpoint-best`
- **Final Model**: `./health-agent-distributed-hub/`

### **Enhanced Training Info:**
```json
{
  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "dataset_name": "aldsouza/health-agent-dataset",
  "global_step": 150,
  "best_eval_loss": 0.234,
  "world_size": 2,
  "batch_size": 1,
  "learning_rate": 5e-5,
  "epoch": 2
}
```

## 🔄 Memory Optimization

### **16GB GPU Optimizations:**
- **4-bit Quantization**: Reduces model memory footprint
- **Double Quantization**: Extra memory savings
- **Gradient Checkpointing**: Trades compute for memory
- **Mixed Precision**: Uses float16 for efficiency
- **Batch Size**: 1 per GPU with gradient accumulation
- **Sequence Length**: Reduced to 1024 for memory efficiency

### **Memory Usage per GPU:**
```
GPU 0: ~6-8GB used / 16GB total (50-60% utilization)
GPU 1: ~6-8GB used / 16GB total (50-60% utilization)
```

## 📊 Performance Benefits

### **Distributed Training Advantages:**
- **2x GPUs**: ~1.8x speedup (due to communication overhead)
- **Memory Efficiency**: Each GPU uses ~50% of available memory
- **Larger Effective Batch**: 1 × 2 GPUs × 8 accumulation = 16
- **Better Convergence**: Larger batches often lead to better results

### **Hub Dataset Benefits:**
- **No Local Storage**: No need to store dataset locally
- **Always Up-to-Date**: Uses latest version from Hub
- **Easy Sharing**: Others can use the same dataset
- **Version Control**: Track dataset changes

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
   **Solution**: Reduce batch size or increase gradient accumulation steps

3. **Communication Errors:**
   ```
   ❌ NCCL error
   ```
   **Solution**: Check GPU connectivity and NCCL configuration

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

# Enable NCCL debug
export NCCL_DEBUG=INFO
```

## 🔧 Advanced Configuration

### **Custom Dataset:**
```bash
python distributed_trainer_hub.py \
    --dataset_name "your-username/your-dataset" \
    --world_size 4 \
    --batch_size 2
```

### **Memory Optimization:**
```bash
python distributed_trainer_hub.py \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 512
```

### **Performance Tuning:**
```bash
python distributed_trainer_hub.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4
```

## 📈 Monitoring Training

### **Key Metrics to Watch:**
- **GPU Utilization**: Should be high (>80%) on all GPUs
- **Memory Usage**: Should be consistent across GPUs
- **Training Speed**: Should scale with number of GPUs
- **Communication Time**: Should be minimal compared to computation

### **Progress Indicators:**
```
Epoch 1: 100%|██████████| 6000/6000 [12:30<00:00, 8.0it/s, loss=0.456]
✅ Eval at step 50 | loss: 0.234
💾 New best model saved at step 50 with loss: 0.234
📊 Epoch 1 average loss: 0.456
```

## 🔗 Integration with Other Scripts

### **Use with Single GPU Training:**
```python
# Update trainer_hub_dataset.py to use same dataset
dataset = load_dataset("aldsouza/health-agent-dataset")
```

### **Use with Evaluation:**
```python
# Load trained model
from peft import PeftModel
model = PeftModel.from_pretrained(model, "./health-agent-distributed-hub")
```

## 📚 File Structure

```
health-agent/
├── distributed_trainer_hub.py      # Main distributed training script
├── launch_distributed_hub.sh       # Linux/Mac launcher
├── launch_distributed_hub.bat      # Windows launcher
├── health-agent-distributed-hub/   # Training output directory
│   ├── checkpoint-best/            # Best model checkpoint
│   ├── checkpoint-50/              # Regular checkpoints
│   ├── training_info.json          # Training information
│   └── final_training_info.json    # Final training info
└── DISTRIBUTED_HUB_TRAINING_GUIDE.md  # This documentation
```

## 🎉 Benefits of Hub Dataset + Distributed Training

1. **Scalability**: Easy to scale to more GPUs
2. **Memory Efficiency**: Better GPU utilization
3. **Cloud Dataset**: No local storage required
4. **Collaboration**: Easy to share datasets and models
5. **Version Control**: Track both dataset and model changes
6. **Production Ready**: Robust error handling and monitoring

## 🔄 Complete Workflow

### **End-to-End Process:**
1. **Create Dataset**: Upload to Hub using `push_dataset_to_hub.py`
2. **Distributed Training**: Use `distributed_trainer_hub.py`
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

### **Basic Distributed Training:**
```bash
./launch_distributed_hub.sh 2
```

### **Custom Configuration:**
```bash
python distributed_trainer_hub.py \
    --world_size 4 \
    --dataset_name "aldsouza/health-agent-dataset" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 10
```

### **Memory-Constrained Setup:**
```bash
python distributed_trainer_hub.py \
    --world_size 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 512
```

This distributed training script with Hub dataset provides the best of both worlds: the power of distributed training and the convenience of cloud-based dataset management!

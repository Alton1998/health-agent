# Kaggle Training Guide

This guide explains how to run the health agent training in Kaggle notebooks with the optimized `kaggle_trainer.py` script.

## 🚀 Quick Start for Kaggle

### **Step 1: Create Kaggle Notebook**
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. **Enable GPU**: Go to Settings → Accelerator → GPU T4 x2 (or P100)

### **Step 2: Copy the Code**
Copy the entire `kaggle_trainer.py` code into a Kaggle notebook cell and run it.

### **Step 3: Run Training**
```python
# In Kaggle notebook, just run:
exec(open('kaggle_trainer.py').read())
```

## 🔧 Kaggle-Specific Optimizations

### **Memory Optimizations:**
- **Reduced epochs**: 3 instead of 5 (fits Kaggle time limits)
- **Smaller batch size**: 1 per GPU (conservative for T4)
- **Reduced sequence length**: 512 instead of 1024
- **Limited dataset**: 10,000 samples max
- **Frequent checkpointing**: Every 25 steps

### **Performance Optimizations:**
- **No multiprocessing**: Single GPU training
- **Mixed precision**: Uses float16 for efficiency
- **Gradient checkpointing**: Saves memory
- **Frequent evaluation**: Every 25 steps

### **Kaggle Compatibility:**
- **num_workers=0**: Avoids multiprocessing issues
- **Smaller test set**: 10% instead of 20%
- **Memory monitoring**: Tracks GPU usage
- **Automatic saving**: Saves to Kaggle output

## 📊 Configuration for Kaggle

```python
# Kaggle-optimized settings
NUM_EPOCHS = 3                    # Reduced for time limits
BATCH_SIZE = 1                    # Conservative for T4 GPU
GRADIENT_ACCUMULATION_STEPS = 4   # Reduced for faster training
MAX_LENGTH = 512                  # Reduced for memory efficiency
EVAL_STEPS = 25                   # More frequent evaluation
SAVE_STEPS = 25                   # More frequent saving
```

## 🎯 Expected Performance

### **Kaggle T4 GPU:**
- **Training time**: ~2-3 hours for 3 epochs
- **Memory usage**: ~8-10GB
- **Effective batch size**: 4 (1 × 4 accumulation)
- **Dataset size**: 10,000 samples

### **Kaggle P100 GPU:**
- **Training time**: ~1-2 hours for 3 epochs
- **Memory usage**: ~6-8GB
- **Better performance**: Faster training

## 📋 Step-by-Step Instructions

### **1. Setup Kaggle Notebook:**
```
1. Go to kaggle.com/code
2. Click "New Notebook"
3. Enable GPU (T4 or P100)
4. Set internet to "On" (for dataset download)
```

### **2. Install Dependencies:**
```python
# First cell - install packages
!pip install transformers datasets torch accelerate
```

### **3. Run Training:**
```python
# Second cell - copy and run kaggle_trainer.py
# (Copy the entire kaggle_trainer.py content here)
```

### **4. Monitor Progress:**
The script will show:
```
🚀 Starting Kaggle Training
🔍 GPU: Tesla T4
🔍 GPU Memory: 15.0GB
🤖 Loading model: aldsouza/health-agent-lora-rkllm-compatible
📊 Loading dataset from Hub: aldsouza/health-agent-dataset
✅ Successfully loaded dataset from Hub
📊 Limited train dataset to 10,000 samples for Kaggle
🔤 Tokenizing datasets...
🤖 Loading pre-trained LoRA model...
📊 Trainable parameters: 4,194,304
📊 Total parameters: 1,500,000,000
📊 Trainable %: 0.28%
🚀 Starting training...
```

## 💾 Downloading Results

### **After Training Completes:**
1. Go to the "Output" tab in Kaggle
2. Find the `health-agent-kaggle-trained` folder
3. Download the entire folder
4. Use in your local environment

### **Output Structure:**
```
health-agent-kaggle-trained/
├── checkpoint-best/              # Best model checkpoint
├── checkpoint-25/                # Regular checkpoints
├── checkpoint-50/
├── training_info.json            # Training information
└── final_training_info.json      # Final training info
```

## 🛠️ Troubleshooting

### **Common Kaggle Issues:**

1. **Out of Memory:**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: The script is already optimized, but you can reduce `BATCH_SIZE` to 1

2. **Dataset Download Fails:**
   ```
   Error loading dataset from Hub
   ```
   **Solution**: Make sure internet is enabled in Kaggle settings

3. **Training Too Slow:**
   ```
   Training is very slow
   ```
   **Solution**: Use P100 GPU instead of T4, or reduce dataset size

4. **Time Limit Exceeded:**
   ```
   Session timeout
   ```
   **Solution**: Reduce `NUM_EPOCHS` to 2 or 1

### **Memory Monitoring:**
The script shows memory usage:
```
🔍 GPU Memory after model loading: Allocated=2.45GB, Reserved=3.12GB
🔍 GPU Memory after moving to device: Allocated=4.23GB, Reserved=5.67GB
🔍 GPU Memory before training: Allocated=4.23GB, Reserved=5.67GB
```

## 📈 Performance Tips

### **For Faster Training:**
1. **Use P100 GPU**: Much faster than T4
2. **Reduce epochs**: 2-3 epochs instead of 5
3. **Smaller dataset**: Already limited to 10k samples
4. **Frequent saving**: Saves every 25 steps

### **For Better Results:**
1. **Use T4 x2**: More memory for larger batches
2. **Increase epochs**: If time allows
3. **Full dataset**: Remove the 10k limit
4. **Longer sequences**: Increase MAX_LENGTH

## 🔄 Using Trained Model

### **Load in Local Environment:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the trained model
model = AutoModelForCausalLM.from_pretrained("./health-agent-kaggle-trained")
tokenizer = AutoTokenizer.from_pretrained("./health-agent-kaggle-trained")

# Use for inference
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📚 Kaggle Notebook Template

### **Complete Notebook Structure:**
```python
# Cell 1: Install dependencies
!pip install transformers datasets torch accelerate

# Cell 2: Import and run training
exec(open('kaggle_trainer.py').read())

# Cell 3: Test the trained model (optional)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./health-agent-kaggle-trained")
tokenizer = AutoTokenizer.from_pretrained("./health-agent-kaggle-trained")
```

## 🎉 Benefits of Kaggle Training

1. **Free GPU**: Use T4 or P100 GPUs for free
2. **No Setup**: No local environment setup needed
3. **Easy Sharing**: Share notebooks with others
4. **Automatic Saving**: Models saved to Kaggle output
5. **Memory Monitoring**: Built-in GPU memory tracking

## ⚠️ Kaggle Limitations

1. **Time Limits**: 9 hours max per session
2. **Memory Limits**: T4 has ~15GB, P100 has ~16GB
3. **Internet**: Must enable for dataset download
4. **Output Size**: Limited output storage

The Kaggle-optimized version is perfect for experimenting with your health agent model without needing local GPU resources!

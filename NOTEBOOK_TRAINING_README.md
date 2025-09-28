# Notebook-Compatible Training Guide

This guide explains how to run training in Jupyter notebooks and interactive environments, which have different requirements than regular Python scripts.

## 🚨 The Problem

When running distributed training in Jupyter notebooks, you might encounter this error:
```
AttributeError: Can't get attribute 'train_worker' on <module '__main__' (built-in)>
```

This happens because:
- Jupyter notebooks use `__main__` as the module name
- Multiprocessing spawn can't find functions in notebook environments
- Distributed training requires proper module structure

## 🔧 Solutions

### **Option 1: Use Notebook-Compatible Script**

Run the notebook-compatible version:
```python
# In Jupyter notebook
exec(open('distributed_trainer_hub_notebook.py').read())
```

Or use the simple launcher:
```bash
python train_notebook.py
```

### **Option 2: Run as Regular Script**

For distributed training, run as a regular Python script:
```bash
python distributed_trainer_hub_simple.py
```

## 📊 Comparison of Versions

| Feature | `distributed_trainer_hub_simple.py` | `distributed_trainer_hub_notebook.py` |
|---------|-------------------------------------|---------------------------------------|
| **Environment** | Command line scripts | Jupyter notebooks |
| **Multi-GPU** | ✅ Full distributed training | ❌ Single GPU only |
| **Multiprocessing** | ✅ Uses mp.spawn | ❌ Single process |
| **Performance** | ✅ 1.8x speedup with 2 GPUs | ❌ Single GPU performance |
| **Memory Usage** | ✅ Distributed across GPUs | ❌ Single GPU memory |
| **Compatibility** | ❌ Not for notebooks | ✅ Works in notebooks |

## 🚀 Quick Start

### **For Jupyter Notebooks:**
```python
# Method 1: Direct execution
exec(open('distributed_trainer_hub_notebook.py').read())

# Method 2: Import and run
from distributed_trainer_hub_notebook import main
main()

# Method 3: Use launcher
!python train_notebook.py
```

### **For Command Line:**
```bash
# Distributed training (recommended)
python distributed_trainer_hub_simple.py

# Or use launcher
./launch_distributed_simple.sh
```

## 🔧 Configuration

Both versions use the same configuration:

```python
# Model and Dataset
MODEL_NAME = "aldsouza/health-agent-lora-rkllm-compatible"
DATASET_NAME = "aldsouza/health-agent-dataset"
OUTPUT_DIR = "./health-agent-distributed-hub"

# Training
NUM_EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 8
```

## 📊 Performance Comparison

### **Distributed Training (Script):**
- **2x 16GB GPUs**: ~1.8x speedup
- **Memory per GPU**: ~6-8GB
- **Effective batch size**: 16
- **Training time**: ~45% reduction

### **Single GPU Training (Notebook):**
- **1x 16GB GPU**: Baseline performance
- **Memory usage**: ~12-14GB
- **Effective batch size**: 8
- **Training time**: Standard

## 🛠️ Troubleshooting

### **Common Issues:**

1. **Multiprocessing Error in Notebook:**
   ```
   AttributeError: Can't get attribute 'train_worker'
   ```
   **Solution**: Use `distributed_trainer_hub_notebook.py`

2. **CUDA Out of Memory:**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use distributed training

3. **Slow Training:**
   ```
   Training is very slow
   ```
   **Solution**: Use distributed training with multiple GPUs

### **Best Practices:**

1. **For Development**: Use notebook version for experimentation
2. **For Production**: Use distributed version for full training
3. **For Large Models**: Always use distributed training
4. **For Quick Tests**: Use notebook version

## 📋 Usage Examples

### **Jupyter Notebook:**
```python
# Cell 1: Setup
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Cell 2: Run training
exec(open('distributed_trainer_hub_notebook.py').read())
```

### **Command Line:**
```bash
# Check environment
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run distributed training
python distributed_trainer_hub_simple.py
```

## 🎯 Recommendations

### **Choose Based on Your Needs:**

1. **Jupyter Notebook Users:**
   - Use `distributed_trainer_hub_notebook.py`
   - Single GPU training
   - Good for experimentation

2. **Command Line Users:**
   - Use `distributed_trainer_hub_simple.py`
   - Multi-GPU training
   - Best performance

3. **Production Training:**
   - Always use distributed version
   - Run as script, not notebook
   - Use multiple GPUs

## 📚 File Structure

```
health-agent/
├── distributed_trainer_hub_simple.py      # Distributed training (scripts)
├── distributed_trainer_hub_notebook.py    # Single GPU training (notebooks)
├── train_notebook.py                      # Notebook launcher
├── launch_distributed_simple.sh           # Script launcher
└── NOTEBOOK_TRAINING_README.md            # This guide
```

## 🔄 Migration Guide

### **From Notebook to Script:**
1. Save your notebook work
2. Run `python distributed_trainer_hub_simple.py`
3. Use the same configuration
4. Get better performance with multiple GPUs

### **From Script to Notebook:**
1. Use `distributed_trainer_hub_notebook.py`
2. Run in notebook environment
3. Single GPU training
4. Good for development and testing

## 🎉 Summary

- **Notebook Version**: Single GPU, notebook-compatible, good for development
- **Script Version**: Multi-GPU, distributed training, best performance
- **Choose based on your environment and needs**
- **Both use the same model and dataset**
- **Configuration is identical between versions**

The notebook version sacrifices performance for compatibility, while the script version provides full distributed training capabilities. Choose the version that best fits your workflow!

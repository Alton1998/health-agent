# Distributed Training Guide for Health Agent

## Overview

This guide explains how to use the `distributed_trainer.py` script for training your health agent model across multiple GPUs using PyTorch's distributed training capabilities.

## What is Distributed Training?

Distributed training is a technique that allows you to train machine learning models across multiple GPUs or machines simultaneously. Instead of training on a single GPU, you can leverage multiple GPUs to:

1. **Scale Training**: Process larger batches or train faster
2. **Handle Large Models**: Distribute model parameters across devices
3. **Process More Data**: Each GPU processes a different subset of data

## Key Concepts

### 1. Process Groups
- **World Size**: Total number of processes (usually equal to number of GPUs)
- **Rank**: Unique identifier for each process (0 to world_size-1)
- **Local Rank**: GPU ID within each machine (usually same as rank for single machine)

### 2. Data Parallelism
- **Distributed Data Parallel (DDP)**: Each GPU gets a copy of the model and processes different data
- **Gradient Synchronization**: Gradients are averaged across all GPUs after each backward pass
- **Distributed Sampler**: Ensures each GPU processes different data samples

### 3. Communication Backends
- **NCCL**: Optimized for NVIDIA GPUs (recommended for GPU training)
- **Gloo**: Cross-platform backend (works on CPU and GPU)

## How the Distributed Script Works

### 1. Initialization (`setup_distributed`)
```python
def setup_distributed(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'  # Master node address
    os.environ['MASTER_PORT'] = '12355'      # Communication port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set GPU for this process
```

**What happens:**
- Each process connects to a master process (rank 0)
- Establishes communication channels between processes
- Assigns each process to a specific GPU

### 2. Model Wrapping (`DistributedDataParallel`)
```python
peft_model = DDP(peft_model, device_ids=[rank], output_device=rank)
```

**What happens:**
- Creates a copy of the model on each GPU
- Wraps the model to handle gradient synchronization
- Automatically averages gradients across all processes

### 3. Distributed Data Loading
```python
train_sampler = DistributedSampler(
    train_dataset, 
    num_replicas=world_size, 
    rank=rank,
    shuffle=True,
    seed=42
)
```

**What happens:**
- Each GPU gets a different subset of data
- No data overlap between processes
- Maintains reproducibility with fixed seeds

### 4. Gradient Synchronization
```python
# DDP automatically handles this after backward pass:
# 1. Compute gradients on each GPU
# 2. Average gradients across all GPUs
# 3. Update model parameters
```

### 5. Evaluation Synchronization
```python
# Gather evaluation losses from all processes
eval_losses_tensor = torch.tensor(eval_losses, device=device)
gathered_losses = [torch.zeros_like(eval_losses_tensor) for _ in range(world_size)]
dist.all_gather(gathered_losses, eval_losses_tensor)
```

**What happens:**
- Each GPU computes evaluation metrics
- Results are gathered and averaged across all processes
- Only rank 0 prints results and saves models

## Key Differences from Single GPU Training

| Aspect | Single GPU | Distributed |
|--------|------------|-------------|
| **Model** | Direct model | `DDP(model)` wrapper |
| **Data** | Regular DataLoader | `DistributedSampler` |
| **Batch Size** | Total batch size | Per-GPU batch size |
| **Gradients** | Direct computation | Synchronized across GPUs |
| **Saving** | Any process | Only rank 0 |
| **Logging** | Any process | Only rank 0 |

## Performance Benefits

### Theoretical Speedup
- **2 GPUs**: ~1.8x speedup (due to communication overhead)
- **4 GPUs**: ~3.2x speedup
- **8 GPUs**: ~6.4x speedup

### Memory Benefits
- Each GPU only needs to store 1/N of the total batch
- Can train with larger effective batch sizes
- Better memory utilization

## Usage Instructions

### 1. Basic Usage
```bash
python distributed_trainer.py --world_size 2
```

### 2. Custom Configuration
```bash
python distributed_trainer.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset_path "./health-agent-dataset" \
    --output_dir "./health-agent-distributed" \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --world_size 4
```

### 3. Using the Launch Script
```bash
bash launch_distributed.sh 4  # Launch with 4 GPUs
```

## Command Line Arguments

### Model and Data
- `--model_name`: Hugging Face model name or path
- `--dataset_path`: Path to your dataset
- `--output_dir`: Directory to save checkpoints

### Training Parameters
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size per GPU (total = batch_size × world_size)
- `--learning_rate`: Learning rate
- `--gradient_accumulation_steps`: Steps to accumulate gradients
- `--max_grad_norm`: Gradient clipping threshold

### LoRA Parameters
- `--lora_r`: LoRA rank
- `--lora_alpha`: LoRA alpha parameter
- `--lora_dropout`: LoRA dropout rate

### Training Control
- `--eval_steps`: Evaluation frequency
- `--save_steps`: Checkpoint saving frequency
- `--log_steps`: Logging frequency
- `--world_size`: Number of GPUs to use

## Best Practices

### 1. Batch Size Scaling
- **Effective batch size** = `batch_size × world_size × gradient_accumulation_steps`
- Start with smaller per-GPU batch sizes and scale up
- Monitor GPU memory usage

### 2. Learning Rate Scaling
- Often need to scale learning rate with batch size
- Common rule: `lr = base_lr × sqrt(world_size)`
- Or use linear scaling: `lr = base_lr × world_size`

### 3. Communication Optimization
- Use NCCL backend for NVIDIA GPUs
- Ensure good network connectivity between GPUs
- Consider gradient compression for large models

### 4. Memory Management
- Use gradient checkpointing for large models
- Monitor memory usage across GPUs
- Consider mixed precision training (already included)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size per GPU
   - Increase gradient accumulation steps
   - Use gradient checkpointing

2. **Communication Errors**
   - Check firewall settings
   - Ensure all GPUs are accessible
   - Try different MASTER_PORT

3. **Slow Training**
   - Check GPU utilization
   - Monitor communication overhead
   - Consider data loading bottlenecks

### Debugging Tips

1. **Check GPU Usage**
   ```bash
   nvidia-smi
   ```

2. **Monitor Process Communication**
   ```bash
   # Check if processes can communicate
   python -c "import torch.distributed as dist; print('DDP available:', dist.is_available())"
   ```

3. **Verify Data Distribution**
   - Check that each GPU processes different data
   - Monitor batch sizes across processes

## Performance Monitoring

### Key Metrics to Watch
- **GPU Utilization**: Should be high (>80%) on all GPUs
- **Memory Usage**: Should be consistent across GPUs
- **Training Speed**: Should scale with number of GPUs
- **Communication Time**: Should be minimal compared to computation

### Logging
- Only rank 0 prints training progress
- All processes contribute to evaluation metrics
- Checkpoint saving is handled by rank 0

## Advanced Features

### 1. Multi-Node Training
For training across multiple machines, modify:
```python
os.environ['MASTER_ADDR'] = 'your-master-node-ip'
os.environ['MASTER_PORT'] = '12355'
```

### 2. Gradient Compression
For large models, consider gradient compression:
```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
peft_model.register_comm_hook(None, default_hooks.fp16_compress_hook)
```

### 3. Dynamic Loss Scaling
The script already includes automatic mixed precision with dynamic loss scaling.

## Conclusion

Distributed training allows you to:
- Train larger models
- Process more data
- Reduce training time
- Better utilize hardware resources

The `distributed_trainer.py` script provides a production-ready implementation that handles all the complexities of distributed training while maintaining the same interface as single-GPU training.

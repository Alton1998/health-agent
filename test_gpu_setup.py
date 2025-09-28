#!/usr/bin/env python3
"""
Simple GPU test script to verify multi-GPU setup works before training.
"""

import torch
import os

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_gpu_setup():
    """Test GPU setup and memory distribution."""
    
    print("🔍 Testing GPU Setup...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPUs")
    
    # Test each GPU
    for i in range(num_gpus):
        try:
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"   Memory: {props.total_memory / 1024**3:.1f}GB")
            
            # Test basic operations
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.mm(test_tensor, test_tensor.t())
            del test_tensor, result
            torch.cuda.empty_cache()
            print(f"   ✅ GPU {i} working correctly")
            
        except Exception as e:
            print(f"   ❌ GPU {i} failed: {e}")
            return False
    
    # Test multi-GPU memory distribution
    if num_gpus > 1:
        print("\n🔍 Testing Multi-GPU Memory Distribution...")
        
        try:
            from transformers import AutoModelForCausalLM
            
            # Load a small model to test device_map
            model_name = "aldsouza/health-agent-lora-rkllm-compatible"
            print(f"📥 Loading model: {model_name}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            print("✅ Model loaded with device_map='auto'")
            
            # Check memory usage on each GPU
            for i in range(num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.1f}GB total")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            print("✅ Multi-GPU setup test passed")
            
        except Exception as e:
            print(f"❌ Multi-GPU test failed: {e}")
            return False
    
    print("\n🎉 All GPU tests passed! Ready for training.")
    return True

if __name__ == "__main__":
    success = test_gpu_setup()
    if success:
        print("\n✅ GPU setup is working correctly. You can now run the training script.")
    else:
        print("\n❌ GPU setup has issues. Please check your configuration.")


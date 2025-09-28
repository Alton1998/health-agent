# Dataset Upload Guide

This guide explains how to upload your health agent dataset to the Hugging Face Hub using the new dedicated upload script.

## 🚀 Quick Start

### **Step 1: Create the Dataset**
```bash
# First, create the dataset locally
python data_set_2.py
```

### **Step 2: Upload to Hub**
```bash
# Option A: Use the Python script directly
python push_dataset_to_hub.py

# Option B: Use the launcher script (Linux/Mac)
chmod +x upload_dataset.sh
./upload_dataset.sh

# Option C: Use the batch file (Windows)
upload_dataset.bat
```

## 📁 File Structure

```
health-agent/
├── data_set_2.py              # Creates the dataset locally
├── push_dataset_to_hub.py     # Uploads dataset to Hub
├── upload_dataset.sh          # Linux/Mac launcher
├── upload_dataset.bat         # Windows launcher
├── setup_hf_token.py          # Authentication helper
└── health-agent-dataset/      # Local dataset (created by data_set_2.py)
```

## 🔧 What Each File Does

### **`push_dataset_to_hub.py`** - Main Upload Script
- ✅ Loads the local dataset from `./health-agent-dataset`
- ✅ Handles Hugging Face authentication
- ✅ Creates repository on the Hub
- ✅ Uploads the dataset
- ✅ Creates a dataset card (README)
- ✅ Provides usage instructions

### **`upload_dataset.sh`** - Linux/Mac Launcher
- ✅ Checks if Python is installed
- ✅ Verifies dataset exists
- ✅ Checks dependencies
- ✅ Runs the upload script

### **`upload_dataset.bat`** - Windows Launcher
- ✅ Same checks as shell script
- ✅ Windows-compatible batch file
- ✅ Pauses for user input

## 🔐 Authentication Setup

### **Method 1: Interactive Login**
```bash
python push_dataset_to_hub.py
# Will prompt for token during execution
```

### **Method 2: Environment Variable**
```bash
# Set token in environment
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Then run upload
python push_dataset_to_hub.py
```

### **Method 3: Use Setup Script**
```bash
python setup_hf_token.py
# Follow the guided setup
```

## 📊 Dataset Information

### **What Gets Uploaded:**
- **Source**: Salesforce function calling dataset (60k samples)
- **Format**: Chat messages with function calling
- **Size**: ~60,000 training examples
- **Use Case**: Health agent fine-tuning

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
      "content": "What's the weather like?"
    },
    {
      "role": "assistant",
      "content": "[{\"name\": \"get_weather\", \"arguments\": {...}}]"
    }
  ]
}
```

## 🎯 Upload Process

### **Step-by-Step:**
1. **Check Dataset**: Verifies local dataset exists
2. **Load Dataset**: Loads from `./health-agent-dataset`
3. **Authenticate**: Handles Hugging Face login
4. **Get Info**: Prompts for repository name and privacy
5. **Create Repo**: Creates repository on Hub
6. **Upload Data**: Pushes dataset to Hub
7. **Create Card**: Adds README with usage instructions

### **Interactive Prompts:**
```
📝 Repository ID (default: your-username/health-agent-dataset): 
🔒 Make repository private? (y/n, default: n): 
```

## 🔒 Privacy Options

### **Public Dataset:**
- ✅ Visible to everyone
- ✅ Can be used by others
- ✅ Contributes to open source
- ❌ Cannot be made private later

### **Private Dataset:**
- ✅ Only you can see it
- ✅ Can be shared with specific users
- ✅ Can be made public later
- ❌ Requires Pro subscription for large datasets

## 📖 Usage After Upload

### **Load Your Dataset:**
```python
from datasets import load_dataset

# Load your uploaded dataset
dataset = load_dataset("your-username/health-agent-dataset")
print(dataset)

# Use in training
train_data = dataset["train"]
```

### **Use in Training Scripts:**
```python
# Update your training scripts to use Hub dataset
dataset = load_dataset("your-username/health-agent-dataset")
# Instead of: dataset = load_from_disk("./health-agent-dataset")
```

## 🛠️ Troubleshooting

### **Common Issues:**

1. **Dataset Not Found:**
   ```
   ❌ Dataset not found at ./health-agent-dataset
   ```
   **Solution**: Run `python data_set_2.py` first

2. **Authentication Error:**
   ```
   ❌ Not logged in to Hugging Face Hub
   ```
   **Solution**: Get token from https://huggingface.co/settings/tokens

3. **Repository Already Exists:**
   ```
   ⚠️ Repository your-username/health-agent-dataset already exists
   ```
   **Solution**: Choose different name or allow overwrite

4. **Permission Denied:**
   ```
   ❌ Error: Permission denied
   ```
   **Solution**: Check token has write permissions

5. **Large Dataset Upload:**
   ```
   ❌ Error: Dataset too large
   ```
   **Solution**: Use Git LFS or split into smaller chunks

### **Debug Mode:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Requirements

### **Python Packages:**
```bash
pip install datasets huggingface_hub
```

### **Hugging Face Account:**
- Create account at https://huggingface.co
- Generate token with write permissions
- Token available at https://huggingface.co/settings/tokens

## 🎉 Success Output

After successful upload, you'll see:
```
🎉 Dataset upload completed successfully!
==================================================
🔗 View your dataset: https://huggingface.co/datasets/your-username/health-agent-dataset
📖 Load in code: dataset = load_dataset('your-username/health-agent-dataset')
🔒 Privacy: Public

💡 Usage Example:
```python
from datasets import load_dataset
dataset = load_dataset('your-username/health-agent-dataset')
print(dataset)
```
```

## 🔄 Workflow Integration

### **Complete Workflow:**
1. **Create Dataset**: `python data_set_2.py`
2. **Upload to Hub**: `python push_dataset_to_hub.py`
3. **Use in Training**: Update training scripts to use Hub dataset
4. **Share**: Make public or share with collaborators

### **Update Training Scripts:**
```python
# Old way (local)
dataset = load_from_disk("./health-agent-dataset")

# New way (Hub)
dataset = load_dataset("your-username/health-agent-dataset")
```

## 📚 Related Files

- `data_set_2.py` - Creates the dataset locally
- `push_dataset_to_hub.py` - Uploads to Hub
- `distributed_trainer.py` - Can use Hub datasets
- `trainer_2.py` - Can be updated to use Hub datasets

## 🔗 Useful Links

- [Hugging Face Hub](https://huggingface.co/datasets)
- [Dataset Documentation](https://huggingface.co/docs/datasets/)
- [Token Management](https://huggingface.co/settings/tokens)
- [Repository Creation](https://huggingface.co/new-dataset)

# Dataset Upload to Hugging Face Hub

This guide explains how to use the updated `data_set_2.py` script to process and upload your health agent dataset to the Hugging Face Hub.

## 🚀 Quick Start

### 1. Setup Hugging Face Authentication
```bash
# Run the setup script first
python setup_hf_token.py
```

### 2. Run the Dataset Script
```bash
# Process and upload dataset
python data_set_2.py
```

## 📋 What the Script Does

### **Data Processing:**
1. **Loads** the Salesforce function calling dataset (60k samples)
2. **Processes** each sample into a chat format with:
   - System prompt with tool definitions
   - User query
   - Assistant response with function calls
3. **Saves** locally to `./health-agent-dataset`

### **Hub Upload:**
1. **Prompts** you to upload to Hugging Face Hub
2. **Asks** for repository name and privacy settings
3. **Uploads** the processed dataset
4. **Provides** usage instructions

## 🔧 Configuration

### **Repository Settings:**
```python
# In data_set_2.py, modify these settings:
REPO_ID = "your-username/health-agent-dataset"  # Change to your username
PRIVATE = False  # Set to True for private dataset
HF_TOKEN = None  # Set your token here or use login()
```

### **Dataset Information:**
- **Source**: Salesforce/xlam-function-calling-60k
- **Size**: ~60,000 samples
- **Format**: Chat messages with function calling
- **Use Case**: Training health agent models

## 📊 Dataset Structure

### **Input Format:**
```json
{
  "query": "What's the weather like?",
  "tools": "[{\"name\": \"get_weather\", \"parameters\": {...}}]",
  "answers": "[{\"name\": \"get_weather\", \"arguments\": {...}}]"
}
```

### **Output Format:**
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

## 🔐 Authentication Methods

### **Method 1: Interactive Login**
```python
from huggingface_hub import login
login()  # Will prompt for token
```

### **Method 2: Token in Code**
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### **Method 3: Environment Variable**
```bash
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

### **Method 4: .env File**
```bash
# Create .env file
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env
```

## 📝 Repository Naming

### **Good Examples:**
- `your-username/health-agent-dataset`
- `your-username/function-calling-health-data`
- `your-username/medical-ai-training-data`

### **Naming Rules:**
- Use lowercase letters and hyphens
- No spaces or special characters
- Be descriptive but concise
- Check if name already exists

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

### **Share with Others:**
```python
# Others can use your dataset
dataset = load_dataset("your-username/health-agent-dataset")
```

## 🛠️ Troubleshooting

### **Common Issues:**

1. **Authentication Error:**
   ```
   ❌ Error: Not authenticated
   ```
   **Solution**: Run `python setup_hf_token.py` and login

2. **Repository Already Exists:**
   ```
   ❌ Error: Repository already exists
   ```
   **Solution**: Choose a different repository name

3. **Permission Denied:**
   ```
   ❌ Error: Permission denied
   ```
   **Solution**: Check your token has write permissions

4. **Large Dataset Upload:**
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

## 📊 Dataset Statistics

After processing, your dataset will have:
- **~60,000 training samples**
- **Chat format** ready for fine-tuning
- **Function calling** examples
- **System prompts** with tool definitions

## 🎯 Next Steps

1. **Upload Dataset**: Run the script and upload to Hub
2. **Update Training Scripts**: Use your Hub dataset in training
3. **Share**: Make it public or share with collaborators
4. **Version**: Create new versions as you improve the data

## 📚 Related Files

- `data_set_2.py` - Main dataset processing and upload script
- `setup_hf_token.py` - Authentication setup helper
- `distributed_trainer.py` - Training script that can use Hub datasets
- `trainer_2.py` - Original training script

## 🔗 Useful Links

- [Hugging Face Hub](https://huggingface.co/datasets)
- [Dataset Documentation](https://huggingface.co/docs/datasets/)
- [Token Management](https://huggingface.co/settings/tokens)
- [Repository Creation](https://huggingface.co/new-dataset)

#!/usr/bin/env python3
"""
Push Health Agent Dataset to Hugging Face Hub
This script loads the locally saved dataset and uploads it to the Hub
"""

import os
import json
from datasets import load_from_disk, Dataset
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def check_dataset_exists():
    """Check if the local dataset exists"""
    dataset_path = "./health-agent-dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("💡 Please run data_set_2.py first to create the dataset")
        return False
    
    print(f"✅ Found dataset at {dataset_path}")
    return True

def load_local_dataset():
    """Load the locally saved dataset"""
    try:
        print("📊 Loading local dataset...")
        dataset = load_from_disk("./health-agent-dataset")
        print(f"✅ Dataset loaded successfully")
        print(f"📋 Dataset info: {dataset}")
        print(f"📊 Total samples: {len(dataset['train'])}")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def setup_hf_auth():
    """Setup Hugging Face authentication"""
    try:
        # Check if already logged in
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Already logged in as: {user_info['name']}")
        return True
    except Exception:
        print("🔐 Not logged in to Hugging Face Hub")
        
        # Try to get token from environment
        token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if token:
            try:
                login(token=token)
                api = HfApi()
                user_info = api.whoami()
                print(f"✅ Logged in with environment token as: {user_info['name']}")
                return True
            except Exception as e:
                print(f"❌ Invalid token in environment: {e}")
        
        # Interactive login
        print("\n📝 To get your Hugging Face token:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Click 'New token'")
        print("3. Give it a name (e.g., 'dataset-upload')")
        print("4. Select 'Write' permissions")
        print("5. Copy the token")
        
        token = input("\n🔑 Enter your Hugging Face token: ").strip()
        if not token:
            print("❌ No token provided")
            return False
        
        try:
            login(token=token)
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ Successfully logged in as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Failed to login: {e}")
            return False

def get_repo_info():
    """Get repository information from user"""
    print("\n📝 Repository Information")
    print("=" * 40)
    
    # Get username
    try:
        api = HfApi()
        user_info = api.whoami()
        default_username = user_info['name']
    except:
        default_username = "your-username"
    
    # Get repository name
    default_repo = f"{default_username}/health-agent-dataset"
    repo_id = input(f"📝 Repository ID (default: {default_repo}): ").strip()
    if not repo_id:
        repo_id = default_repo
    
    # Validate repository name
    if "/" not in repo_id:
        print("❌ Repository ID must be in format 'username/dataset-name'")
        return None, None
    
    # Get privacy setting
    private_choice = input("🔒 Make repository private? (y/n, default: n): ").lower().strip()
    private = private_choice in ['y', 'yes']
    
    return repo_id, private

def create_dataset_repo(repo_id, private=False):
    """Create the dataset repository on Hugging Face Hub"""
    try:
        api = HfApi()
        
        # Check if repo already exists
        try:
            repo_info = api.repo_info(repo_id, repo_type="dataset")
            print(f"⚠️ Repository {repo_id} already exists")
            overwrite = input("🔄 Do you want to overwrite it? (y/n): ").lower().strip()
            if overwrite not in ['y', 'yes']:
                return False
        except RepositoryNotFoundError:
            # Repository doesn't exist, create it
            print(f"🚀 Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            print(f"✅ Repository created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False

def push_dataset_to_hub(dataset, repo_id, private=False):
    """Push the dataset to Hugging Face Hub"""
    try:
        print(f"🚀 Pushing dataset to hub: {repo_id}")
        print("⏳ This may take a few minutes...")
        
        # Push the dataset
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message="Upload health agent dataset"
        )
        
        print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing dataset: {e}")
        return False

def create_dataset_card(repo_id):
    """Create a README for the dataset"""
    readme_content = f"""---
license: mit
tags:
- health
- medical
- function-calling
- ai-assistant
- healthcare
size_categories:
- 10K<n<100K
---

# Health Agent Dataset

This dataset contains function calling examples for training health-related AI assistants.

## Dataset Description

This dataset is derived from the Salesforce function calling dataset and has been processed specifically for health agent training. It contains examples of function calling in medical and health-related contexts.

## Dataset Structure

The dataset contains approximately 60,000 examples in chat format with the following structure:

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "You are an intelligent AI assistant that uses available tools..."
    }},
    {{
      "role": "user",
      "content": "User query here"
    }},
    {{
      "role": "assistant", 
      "content": "[{{\\"name\\": \\"function_name\\", \\"arguments\\": {{...}}}}]"
    }}
  ]
}}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Use in training
train_data = dataset["train"]
```

## Citation

If you use this dataset, please cite the original Salesforce dataset:

```bibtex
@dataset{{salesforce_function_calling,
  title={{Salesforce Function Calling Dataset}},
  author={{Salesforce}},
  year={{2024}},
  url={{https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k}}
}}
```

## License

This dataset is released under the MIT License.
"""
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card"
        )
        print("✅ Dataset card created")
    except Exception as e:
        print(f"⚠️ Could not create dataset card: {e}")

def main():
    """Main function to push dataset to Hub"""
    print("🚀 Health Agent Dataset Upload to Hugging Face Hub")
    print("=" * 60)
    
    # Step 1: Check if dataset exists
    if not check_dataset_exists():
        return
    
    # Step 2: Load the dataset
    dataset = load_local_dataset()
    if dataset is None:
        return
    
    # Step 3: Setup authentication
    if not setup_hf_auth():
        return
    
    # Step 4: Get repository information
    repo_id, private = get_repo_info()
    if repo_id is None:
        return
    
    # Step 5: Create repository
    if not create_dataset_repo(repo_id, private):
        return
    
    # Step 6: Push dataset
    if not push_dataset_to_hub(dataset, repo_id, private):
        return
    
    # Step 7: Create dataset card
    create_dataset_card(repo_id)
    
    # Success message
    print("\n🎉 Dataset upload completed successfully!")
    print("=" * 50)
    print(f"🔗 View your dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"📖 Load in code: dataset = load_dataset('{repo_id}')")
    print(f"🔒 Privacy: {'Private' if private else 'Public'}")
    
    # Usage example
    print("\n💡 Usage Example:")
    print("```python")
    print("from datasets import load_dataset")
    print(f"dataset = load_dataset('{repo_id}')")
    print("print(dataset)")
    print("```")

if __name__ == "__main__":
    main()

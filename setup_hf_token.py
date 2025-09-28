#!/usr/bin/env python3
"""
Setup script for Hugging Face Hub authentication
Run this before using data_set_2.py to push datasets
"""

import os
from huggingface_hub import login, HfApi

def setup_hf_token():
    """Setup Hugging Face token for dataset uploads"""
    
    print("🔐 Hugging Face Hub Setup")
    print("=" * 50)
    
    # Check if already logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Already logged in as: {user_info['name']}")
        print(f"📧 Email: {user_info.get('email', 'Not provided')}")
        
        choice = input("\n🤔 Do you want to use a different account? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("✅ Using current account")
            return True
            
    except Exception:
        print("❌ Not logged in to Hugging Face Hub")
    
    print("\n📝 To get your Hugging Face token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'dataset-upload')")
    print("4. Select 'Write' permissions")
    print("5. Copy the token")
    
    print("\n🔑 Enter your Hugging Face token:")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    try:
        # Test the token
        login(token=token)
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Successfully logged in as: {user_info['name']}")
        
        # Save token to environment (optional)
        save_env = input("\n💾 Save token to environment variable? (y/n): ").lower().strip()
        if save_env in ['y', 'yes']:
            # Add to .env file
            env_file = ".env"
            with open(env_file, "a") as f:
                f.write(f"\nHUGGINGFACE_HUB_TOKEN={token}\n")
            print(f"✅ Token saved to {env_file}")
            print("💡 You can now use: export HUGGINGFACE_HUB_TOKEN=your_token")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to login: {e}")
        return False

def check_repo_name():
    """Help user choose a good repository name"""
    
    print("\n📝 Repository Naming Guidelines:")
    print("=" * 40)
    print("✅ Good names:")
    print("   - your-username/health-agent-dataset")
    print("   - your-username/function-calling-health-data")
    print("   - your-username/medical-ai-training-data")
    print("\n❌ Avoid:")
    print("   - Spaces or special characters")
    print("   - Very long names")
    print("   - Names that already exist")
    
    print("\n💡 Tips:")
    print("- Use lowercase and hyphens")
    print("- Be descriptive but concise")
    print("- Check if name exists at: https://huggingface.co/datasets/your-username/your-dataset-name")

if __name__ == "__main__":
    print("🚀 Hugging Face Dataset Upload Setup")
    print("=" * 50)
    
    # Setup token
    if setup_hf_token():
        print("\n🎉 Setup complete!")
        
        # Show naming guidelines
        check_repo_name()
        
        print("\n📋 Next steps:")
        print("1. Run: python data_set_2.py")
        print("2. Choose your repository name")
        print("3. Decide if you want it private or public")
        print("4. Wait for upload to complete")
        
    else:
        print("\n❌ Setup failed. Please try again.")

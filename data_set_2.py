import json

from datasets import load_dataset

from huggingface_hub import login

# login("")


def preprocess(data):
    tools = json.loads(data["tools"])
    tool_string = json.dumps(tools, indent=2)
    data_entry = dict()
    data_entry["messages"] = list()
    system_prompt = dict()
    system_prompt["role"] = "system"
    system_prompt["content"] = (
        "You are an intelligent AI assistant that uses available tools (functions) to help users achieve their goals. Your job is to understand the user's intent, identify missing information if needed, and then select and call the most appropriate function(s) to solve the task."
        "\n\n# Rules:\n"
        "- ALWAYS use the tools provided to answer the user's request, unless explicitly told not to.\n"
        "- Ask clarifying questions ONLY if the user's request is ambiguous or lacks required input parameters.\n"
        "- If multiple tools are needed, use them in sequence.\n"
        "- DO NOT make up data or assume values — request any missing input clearly.\n"
        "\n# Output Format:\n"
        "- Respond using a JSON list of function calls in the following format:\n"
        "  [\n"
        "    {\n"
        "      \"name\": \"function_name\",\n"
        "      \"arguments\": {\n"
        "        \"param1\": \"value1\",\n"
        "        \"param2\": \"value2\"\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "- Only include the functions needed to complete the task.\n"
        "- If no function is needed or the input is unclear, ask a clarifying question instead of guessing.\n"
        "- Do NOT respond with explanations or natural language outside the JSON block unless explicitly instructed.\n"
        "Following are the tools provided to you:\n"
        f"{tool_string}"
    )
    user_prompt = dict()
    user_prompt["role"] = "user"
    user_prompt["content"] = data["query"]
    assistant_answer = dict()
    assistant_answer["role"] = "assistant"
    function_call = json.loads(data["answers"])
    assistant_answer["content"] = json.dumps(function_call, indent=2)
    data_entry["messages"].append(system_prompt)
    data_entry["messages"].append(user_prompt)
    data_entry["messages"].append(assistant_answer)
    return data_entry


def push_to_hub(dataset, repo_id, token=None, private=False):
    """
    Push dataset to Hugging Face Hub
    
    Args:
        dataset: The processed dataset to upload
        repo_id: Repository ID (e.g., "username/dataset-name")
        token: Hugging Face token (optional, will use login if not provided)
        private: Whether to make the repository private
    """
    try:
        # Login to Hugging Face Hub
        if token:
            login(token=token)
        else:
            # Check if already logged in
            api = HfApi()
            try:
                api.whoami()
                print("✅ Already logged in to Hugging Face Hub")
            except Exception:
                print("🔐 Please login to Hugging Face Hub:")
                login()
        
        # Push dataset to hub
        print(f"🚀 Pushing dataset to hub: {repo_id}")
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token
        )
        print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")
        return False
    
    return True

def main():
    """Main function to process and push dataset"""
    
    # Configuration
    REPO_ID = "your-username/health-agent-dataset"  # Change this to your username
    PRIVATE = False  # Set to True if you want a private dataset
    HF_TOKEN = None  # Set your token here or use login() function
    
    print("📊 Loading dataset from Salesforce/xlam-function-calling-60k...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k")
    
    print("🔄 Processing dataset...")
    processed_dataset = ds.map(preprocess)
    
    print("📋 Dataset info:")
    print(processed_dataset)
    print(f"📊 Total samples: {len(processed_dataset['train'])}")
    
    # Save locally first
    print("💾 Saving dataset locally...")
    processed_dataset.save_to_disk("./health-agent-dataset")
    print("✅ Dataset saved locally to ./health-agent-dataset")
    
    # Ask user if they want to push to hub
    push_to_hub_choice = input("\n🤔 Do you want to push this dataset to Hugging Face Hub? (y/n): ").lower().strip()
    
    if push_to_hub_choice in ['y', 'yes']:
        # Get repository ID from user
        repo_id = input(f"📝 Enter repository ID (default: {REPO_ID}): ").strip()
        if not repo_id:
            repo_id = REPO_ID
        
        # Ask about privacy
        private_choice = input("🔒 Make repository private? (y/n, default: n): ").lower().strip()
        private = private_choice in ['y', 'yes']
        
        # Push to hub
        success = push_to_hub(processed_dataset, repo_id, HF_TOKEN, private)
        
        if success:
            print(f"\n🎉 Dataset successfully uploaded!")
            print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
            print(f"📖 Usage: dataset = load_dataset('{repo_id}')")
        else:
            print("\n❌ Failed to upload dataset. Check the error messages above.")
    else:
        print("📁 Dataset saved locally only.")

if __name__ == "__main__":
    main()


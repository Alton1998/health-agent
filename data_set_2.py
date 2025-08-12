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


ds = load_dataset("Salesforce/xlam-function-calling-60k")

processed_dataset = ds.map(preprocess)

print(processed_dataset)

processed_dataset.save_to_disk("./health-agent-dataset")


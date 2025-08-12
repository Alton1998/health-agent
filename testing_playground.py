# Load model directly
import json
import sys

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",device_map=device)

# with open("./nad_test_dataset.json","r",encoding="utf-8") as file:
# 	data = json.load(file)
# 	entry = data[0]
# 	messages = entry

# messages = messages["messages"][:2]
messages_3 = [
    {
        "role": "system",
        "content": """You are a medical AI assistant. Your task is to understand user requests, identify if any required arguments are missing, and respond by invoking one of the available tools.

<tools>
[
  {
    "name": "request_missing_argument",
    "description": "Ask the user for a required argument to complete a task",
    "parameters": {
      "type": "object",
      "properties": {
        "missing_field": {
          "type": "string",
          "description": "Name of the missing argument (e.g., 'appointment_id')"
        }
      },
      "required": ["missing_field"]
    }
  },
  {
    "name": "book_appointment",
    "description": "Schedule an appointment with a doctor using doctor name and appointment time",
    "parameters": {
      "type": "object",
      "properties": {
        "doctor_name": { "type": "string" },
        "appointment_time": { "type": "string" }
      },
      "required": ["doctor_name", "appointment_time"]
    }
  },
  {
    "name": "cancel_appointment",
    "description": "Cancel an appointment using its unique ID",
    "parameters": {
      "type": "object",
      "properties": {
        "appointment_id": { "type": "string" }
      },
      "required": ["appointment_id"]
    }
  }
]
</tools>

Wrap your response using <tool_call>...</tool_call>. Do not respond in natural language.
"""
    },
    {
        "role": "user",
        "content": "Please cancel my appointment. It’s ID 54789."
    }
]

inputs = tokenizer.apply_chat_template(
	messages_3,
	add_generation_prompt=False,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=4096)
print(tokenizer.decode(outputs[0]))
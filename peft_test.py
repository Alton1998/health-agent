import json
import torch
from regex import regex
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(11)

# --- Base model and adapter ---
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # e.g. "meta-llama/Llama-2-7b-hf"
adapter_model_name = "aldsouza/health-agent-adpater-only"     # PEFT adapter repo

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
).to("cuda")

# Load adapter on top of base model
model = PeftModel.from_pretrained(base_model, adapter_model_name).to("cuda")
model.eval()

medical_tools = [
    {
        "name": "symptom_checker",
        "description": "Analyze symptoms and provide possible conditions.",
        "parameters": {
            "symptoms": {
                "description": "List of symptoms reported by the patient.",
                "type": "list[str]",
                "default": ["headache", "fever"]
            }
        }
    },
    {
        "name": "medication_lookup",
        "description": "Look up details about a medication by its name.",
        "parameters": {
            "medication_name": {
                "description": "Name of the medication to look up.",
                "type": "str",
                "default": "Aspirin"
            }
        }
    },
    {
        "name": "book_appointment",
        "description": "Schedule a medical appointment with a doctor.",
        "parameters": {
            "patient_name": {
                "description": "Name of the patient.",
                "type": "str",
                "default": "John Doe"
            },
            "doctor_specialty": {
                "description": "Specialty of the doctor to book.",
                "type": "str",
                "default": "general practitioner"
            },
            "date": {
                "description": "Preferred date of appointment (YYYY-MM-DD).",
                "type": "str",
                "default": "2025-08-20"
            }
        }
    },
    {
        "name": "get_lab_results",
        "description": "Retrieve lab test results for a patient by test ID.",
        "parameters": {
            "patient_id": {
                "description": "Unique patient identifier.",
                "type": "str",
                "default": "123456"
            },
            "test_id": {
                "description": "Lab test identifier.",
                "type": "str",
                "default": "cbc"
            }
        }
    },
    {
        "name": "request_missing_info",
        "description": "Ask the user for missing or incomplete information needed to fulfill their request.",
        "parameters": {
            "missing_fields": {
                "description": "List of missing required fields to be clarified by the user.",
                "type": "list[str]",
                "default": []
            },
            "context": {
                "description": "Optional context or explanation to help the user provide the missing information.",
                "type": "str",
                "default": ""
            }
        }
    },
    {
        "name": "medical_device_info",
        "description": "Retrieve detailed information about a medical device by its name or model number.",
        "parameters": {
            "device_name": {
                "description": "The name or model number of the medical device to look up.",
                "type": "str",
                "default": "Blood Pressure Monitor"
            }
        }
    }, {
        "name": "record_blood_pressure",
        "description": "Record a patient's blood pressure reading with systolic, diastolic, and pulse rate values.",
        "parameters": {
            "patient_id": {
                "description": "Unique identifier of the patient.",
                "type": "str",
                "default": "123456"
            },
            "systolic": {
                "description": "Systolic blood pressure value (mmHg).",
                "type": "int",
                "default": 120
            },
            "diastolic": {
                "description": "Diastolic blood pressure value (mmHg).",
                "type": "int",
                "default": 80
            },
            "pulse_rate": {
                "description": "Pulse rate in beats per minute.",
                "type": "int",
                "default": 70
            },
            "measurement_time": {
                "description": "Timestamp of the measurement (YYYY-MM-DD HH:MM).",
                "type": "str",
                "default": "2025-08-12 09:00"
            }
        }
    }, {
        "name": "start_blood_pressure_test",
        "description": "Initiate a blood pressure measurement test for a patient using a connected device.",
        "parameters": {
            "patient_id": {
                "description": "Unique identifier of the patient.",
                "type": "str",
                "default": "123456"
            },
            "device_id": {
                "description": "Identifier or model of the blood pressure measuring device.",
                "type": "str",
                "default": "BP-Device-001"
            }
        }
    }
]
# Compose the system prompt embedding the tools JSON
system_prompt = f"""
You are an intelligent AI assistant that uses available tools (functions) to help users achieve their medical-related goals. Your job is to understand the user's intent, identify missing information if needed, and then select and call the most appropriate function(s) to solve the task.

# Rules:
- ALWAYS use the tools provided to answer the user's request, unless explicitly told not to.
- Ask clarifying questions ONLY if the user's request is ambiguous or lacks required input parameters.
- If multiple tools are needed, use them in sequence.
- DO NOT make up data or assume values — request any missing input clearly.

# Output Format:
- Respond using a JSON list of function calls in the following format:
  [
    {{
      "name": "function_name",
      "arguments": {{
        "param1": "value1",
        "param2": "value2"
      }}
  ]
- Only include the functions needed to complete the task.
- If no function is needed or the input is unclear, ask a clarifying question instead of guessing.
- Do NOT respond with explanations or natural language outside the JSON block unless explicitly instructed.

Following are the tools provided to you:
{json.dumps(medical_tools, indent=2)}
"""

# --- Prompts (keep as-is) ---
messages_list = [
    [
        {"content": system_prompt, "role": "system"},
        {"content": (
            "I have a headache and mild fever. What could be the possible conditions? "
            "Also, lookup medication details for 'Ibuprofen'. "
            "Please book an appointment for patient 'Alice Smith' with a neurologist on 2025-09-01."
        ), "role": "user"}
    ],
    [
        {"content": system_prompt, "role": "system"},
        {"content": (
            "My mother has chest pain and shortness of breath. "
            "Can you analyze her symptoms? "
            "Also, please look up information about 'Nitroglycerin' medication. "
            "Finally, get lab results for patient ID '987654' for the test 'lipid_panel'."
        ), "role": "user"}
    ],

]

# --- Generation and regex parsing ---
pattern = r'''
\{
\s*"name"\s*:\s*"([^"]+)"\s*,
\s*"arguments"\s*:\s*(\{       
(?:[^{}]++ | (?2))*?           
\})
\s*\}
'''

for idx, messages in enumerate(messages_list, 1):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
        )

    response = tokenizer.decode(outputs[0])
    print(f"\n--- Prompt {idx} ---")
    print(response)

    matches = regex.findall(pattern, response, regex.VERBOSE)
    for i, (func_name, args_json) in enumerate(matches, 1):
        print(f"Function {i}: {func_name}")
        print("Arguments:")
        print(args_json)
        print("-" * 40)

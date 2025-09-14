import json
import time

import torch
from regex import regex
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(11)
# model_name = "aldsouza/health-agent"
# model_name = "aldsouza/health-agent-merged-fp16"
model_name = "aldsouza/health-agent-lora-rkllm-compatible"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

health_management_tools = [
    # Schedule Management Tools
    {
        "name": "schedule_auto_type_from_name",
        "description": "Automatically determine the type of schedule item based on the name provided.",
        "parameters": {
            "name": {
                "description": "Name of the schedule item to auto-type.",
                "type": "str",
                "default": "Morning Medication"
            }
        }
    },
    {
        "name": "schedule_validate_time_for_test",
        "description": "Validate if the specified time is appropriate for a medical test.",
        "parameters": {
            "test_type": {
                "description": "Type of medical test to validate timing for.",
                "type": "str",
                "default": "blood_pressure"
            },
            "proposed_time": {
                "description": "Proposed time for the test in HH:MM format.",
                "type": "str",
                "default": "09:00"
            }
        }
    },
    {
        "name": "schedule_create_routine",
        "description": "Create a new health routine with specified activities and timing.",
        "parameters": {
            "routine_name": {
                "description": "Name of the routine to create.",
                "type": "str",
                "default": "Morning Health Check"
            },
            "activities": {
                "description": "List of activities to include in the routine.",
                "type": "list[str]",
                "default": ["blood_pressure", "medication", "exercise"]
            },
            "schedule_times": {
                "description": "List of times for each activity in HH:MM format.",
                "type": "list[str]",
                "default": ["08:00", "08:30", "09:00"]
            }
        }
    },
    {
        "name": "schedule_edit_routine",
        "description": "Edit an existing health routine with new parameters.",
        "parameters": {
            "routine_id": {
                "description": "ID of the routine to edit.",
                "type": "str",
                "default": "routine_001"
            },
            "new_activities": {
                "description": "Updated list of activities for the routine.",
                "type": "list[str]",
                "default": ["blood_pressure", "medication", "exercise", "hydration"]
            },
            "new_times": {
                "description": "Updated list of times for each activity.",
                "type": "list[str]",
                "default": ["08:00", "08:30", "09:00", "10:00"]
            }
        }
    },
    {
        "name": "schedule_delete_routine",
        "description": "Delete a health routine by its ID.",
        "parameters": {
            "routine_id": {
                "description": "ID of the routine to delete.",
                "type": "str",
                "default": "routine_001"
            }
        }
    },
    {
        "name": "schedule_list_routines",
        "description": "List all available health routines for the patient.",
        "parameters": {
            "patient_id": {
                "description": "ID of the patient to list routines for.",
                "type": "str",
                "default": "patient_001"
            }
        }
    },
    {
        "name": "schedule_update_meal_times",
        "description": "Update meal times for medication scheduling optimization.",
        "parameters": {
            "breakfast_time": {
                "description": "Breakfast time in HH:MM format.",
                "type": "str",
                "default": "08:00"
            },
            "lunch_time": {
                "description": "Lunch time in HH:MM format.",
                "type": "str",
                "default": "12:30"
            },
            "dinner_time": {
                "description": "Dinner time in HH:MM format.",
                "type": "str",
                "default": "18:00"
            }
        }
    },
    {
        "name": "schedule_list_medications",
        "description": "List all medications and their scheduled times for a patient.",
        "parameters": {
            "patient_id": {
                "description": "ID of the patient to list medications for.",
                "type": "str",
                "default": "patient_001"
            }
        }
    },

    # Appointment Management Tools
    {
        "name": "appointment_get_slots",
        "description": "Get available appointment slots for a specific date and provider.",
        "parameters": {
            "provider_id": {
                "description": "ID of the healthcare provider.",
                "type": "str",
                "default": "provider_001"
            },
            "date": {
                "description": "Date to check for available slots (YYYY-MM-DD format).",
                "type": "str",
                "default": "2024-01-15"
            },
            "appointment_type": {
                "description": "Type of appointment needed.",
                "type": "str",
                "default": "consultation"
            }
        }
    },
    {
        "name": "appointment_create",
        "description": "Create a new appointment with a healthcare provider.",
        "parameters": {
            "patient_id": {
                "description": "ID of the patient scheduling the appointment.",
                "type": "str",
                "default": "patient_001"
            },
            "provider_id": {
                "description": "ID of the healthcare provider.",
                "type": "str",
                "default": "provider_001"
            },
            "date_time": {
                "description": "Date and time for the appointment (YYYY-MM-DD HH:MM format).",
                "type": "str",
                "default": "2024-01-15 10:00"
            },
            "appointment_type": {
                "description": "Type of appointment.",
                "type": "str",
                "default": "consultation"
            }
        }
    },
    {
        "name": "appointment_list_upcoming",
        "description": "List all upcoming appointments for a patient.",
        "parameters": {
            "patient_id": {
                "description": "ID of the patient to list appointments for.",
                "type": "str",
                "default": "patient_001"
            },
            "days_ahead": {
                "description": "Number of days ahead to look for appointments.",
                "type": "int",
                "default": 30
            }
        }
    },
    {
        "name": "appointment_update",
        "description": "Update an existing appointment with new details.",
        "parameters": {
            "appointment_id": {
                "description": "ID of the appointment to update.",
                "type": "str",
                "default": "appt_001"
            },
            "new_date_time": {
                "description": "New date and time for the appointment.",
                "type": "str",
                "default": "2024-01-15 11:00"
            },
            "new_type": {
                "description": "New type of appointment.",
                "type": "str",
                "default": "follow_up"
            }
        }
    },
    {
        "name": "appointment_cancel",
        "description": "Cancel an existing appointment.",
        "parameters": {
            "appointment_id": {
                "description": "ID of the appointment to cancel.",
                "type": "str",
                "default": "appt_001"
            },
            "reason": {
                "description": "Reason for cancellation.",
                "type": "str",
                "default": "patient_request"
            }
        }
    },

    # Communication Tools
    {
        "name": "message_send",
        "description": "Send a message to a healthcare provider or care team member.",
        "parameters": {
            "recipient_id": {
                "description": "ID of the message recipient.",
                "type": "str",
                "default": "provider_001"
            },
            "message_content": {
                "description": "Content of the message to send.",
                "type": "str",
                "default": "Patient reports feeling better after medication adjustment."
            },
            "priority": {
                "description": "Priority level of the message (low, medium, high, urgent).",
                "type": "str",
                "default": "medium"
            }
        }
    },
    {
        "name": "message_attach",
        "description": "Attach a file or document to a message.",
        "parameters": {
            "message_id": {
                "description": "ID of the message to attach file to.",
                "type": "str",
                "default": "msg_001"
            },
            "file_path": {
                "description": "Path to the file to attach.",
                "type": "str",
                "default": "/documents/health_report.pdf"
            },
            "file_type": {
                "description": "Type of file being attached.",
                "type": "str",
                "default": "pdf"
            }
        }
    },
    {
        "name": "careteam_get",
        "description": "Get information about the patient's care team members.",
        "parameters": {
            "patient_id": {
                "description": "ID of the patient to get care team for.",
                "type": "str",
                "default": "patient_001"
            }
        }
    },

    # Camera and Documentation Tools
    {
        "name": "camera_device_open_capture",
        "description": "Open device camera and capture a photo for medical documentation.",
        "parameters": {
            "camera_type": {
                "description": "Type of camera to use (front, back, external).",
                "type": "str",
                "default": "back"
            },
            "capture_purpose": {
                "description": "Purpose of the photo capture (wound_documentation, medication_verification, etc.).",
                "type": "str",
                "default": "wound_documentation"
            }
        }
    },
    {
        "name": "camera_five_in_one_open_capture",
        "description": "Open specialized five-in-one medical camera and capture comprehensive medical images.",
        "parameters": {
            "capture_mode": {
                "description": "Capture mode for the five-in-one camera (full_exam, specific_area, etc.).",
                "type": "str",
                "default": "full_exam"
            },
            "patient_id": {
                "description": "ID of the patient for the capture.",
                "type": "str",
                "default": "patient_001"
            }
        }
    }
]

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
{json.dumps(health_management_tools, indent=2)}
"""

# Example prompt using the medical tools
messages = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "I have a headache and mild fever. What could be the possible conditions? "
            "Also, lookup medication details for 'Ibuprofen'. "
            "Please book an appointment for patient 'Alice Smith' with a neurologist on 2025-09-01."
        ),
        "role": "user"
    }
]
different_user_prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "My mother has chest pain and shortness of breath. "
            "Can you analyze her symptoms? "
            "Also, please look up information about 'Nitroglycerin' medication. "
            "Finally, get lab results for patient ID '987654' for the test 'lipid_panel'."
        ),
        "role": "user"
    }
]

missing_info_user_prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "I need to book a medical appointment. "
            "Please help me schedule it with a cardiologist."
        ),
        "role": "user"
    }
]
medical_device_prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "Can you provide detailed information about the 'Omron BP786N' blood pressure monitor which is a medical device? "
            "Also, my father is experiencing dizziness and blurred vision — can you analyze these symptoms?"
        ),
        "role": "user"
    }
]
blood_pressure_prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "Record a blood pressure reading for patient ID '987654': systolic 135 mmHg, diastolic 85 mmHg, pulse rate 72 bpm, "
            "measured today at 10:30 AM."
        ),
        "role": "user"
    }
]

start_bp_test_prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "Please start a blood pressure test for patient ID '987654' using device 'BP-Device-001'."
        ),
        "role": "user"
    }
]

health_management_prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "Create a morning health routine for patient_001 that includes blood pressure monitoring at 8:00 AM, morning medication at 8:30 AM, and light exercise at 9:00 AM. Then schedule a follow-up appointment with Dr. Smith (provider_001) for next Tuesday at 2:00 PM to discuss the patient's recent health improvements. Also send a message to the care team letting them know the new routine has been established and attach the patient's latest health report from last week."        ),
        "role": "user"
    }
]

inputs = tokenizer.apply_chat_template(
    health_management_prompt,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.7,
    )

    response = tokenizer.decode(outputs[0])

    print(tokenizer.decode(outputs[0]))
    end = time.time()
    print(f"Response:{end-start}")

# inputs = tokenizer.apply_chat_template(
#     different_user_prompt,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=4096,
#         temperature=0.7,
#     )
#
#     response = tokenizer.decode(outputs[0])
#     pattern = r'''
#     \{                      # Opening brace of the function block
#     \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
#     \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
#     (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
#     \})                                # Closing brace of arguments
#     \s*\}                             # Closing brace of the function block
#     '''
#
#     matches = regex.findall(pattern, response, regex.VERBOSE)
#
#     for i, (func_name, args_json) in enumerate(matches, 1):
#         print(f"Function {i}: {func_name}")
#         print("Arguments:")
#         print(args_json)
#         print("-" * 40)
#
#     print(tokenizer.decode(outputs[0]))
#
# inputs = tokenizer.apply_chat_template(
#     missing_info_user_prompt,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=4096,
#         temperature=0.7,
#     )
#
#     response = tokenizer.decode(outputs[0])
#
#     print(tokenizer.decode(outputs[0]))
#
# inputs = tokenizer.apply_chat_template(
#     start_bp_test_prompt,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=4096,
#         temperature=0.7,
#     )
#
#     response = tokenizer.decode(outputs[0])
#
#     print(tokenizer.decode(outputs[0]))
#
# inputs = tokenizer.apply_chat_template(
#     medical_device_prompt,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=4096,
#         temperature=0.7,
#     )
#
#     response = tokenizer.decode(outputs[0])
#
#     print(tokenizer.decode(outputs[0]))

import json
import os
import pickle
from datetime import datetime, timedelta, time as time_1
from threading import Thread
from typing import TypedDict, Dict, List, Any

import pytz
import torch
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_community.tools import TavilySearchResults
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from regex import regex
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from tzlocal import get_localzone
import pyttsx3
import speech_recognition as sr

r = sr.Recognizer()
def speak_text(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

load_dotenv()

torch.manual_seed(11)
model_name = "aldsouza/health-agent"
pattern = r'''
    \{                      # Opening brace of the function block
    \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
    \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
    (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
    \})                                # Closing brace of arguments
    \s*\}                             # Closing brace of the function block
    '''

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
# model_1 = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",torch_dtype=torch.float16).to("cuda")

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
SCOPES = ['https://www.googleapis.com/auth/calendar']

def symptom_checker(kwargs):
    print(f"Checking diseases for following symptoms on the web:")
    symptoms = kwargs.get("symptoms",[])
    print(symptoms)
    for i, arg in enumerate(symptoms):
        print(f"{i}. {arg}")
    results = TavilySearchResults()
    information = ""
    for result in results.invoke(f"What causes {''.join(symptoms)}"):
        information = information + result["content"] + "\n"
    return {
        "status":200,
        "message":information
    }

def medication_lookup(kwargs):
    medication_name = kwargs.get("medication_name")
    print(f"Looking up the web for information on {medication_name}....")
    results = TavilySearchResults()
    information = ""
    for result in results.invoke(f"What is {medication_name}?"):
        information = information + result["content"] + "\n"
    return {
            "status": 200,
            "message": information
        }


def create_google_calendar_meeting(
        summary: str,
        start_datetime: str,
        end_datetime: str,
        attendees_emails: list,
        timezone: str = 'America/Chicago'
):
    """
    Creates a Google Calendar event.

    Args:
        summary (str): Event title.
        start_datetime (str): Start datetime in ISO format, e.g., "2025-08-18T10:00:00-06:00".
        end_datetime (str): End datetime in ISO format.
        attendees_emails (list): List of attendee emails.
        timezone (str): Timezone string, default 'America/Chicago'.
    """

    creds = None
    # Load saved credentials if available
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Authenticate if necessary
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('calendar', 'v3', credentials=creds)

    event = {
        'summary': summary,
        'location': 'Virtual / Google Meet',
        'description': f'{summary} meeting.',
        'start': {'dateTime': start_datetime, 'timeZone': timezone},
        'end': {'dateTime': end_datetime, 'timeZone': timezone},
        'attendees': [{'email': email} for email in attendees_emails],
        'reminders': {'useDefault': True},
    }

    created_event = service.events().insert(
        calendarId='primary', body=event, sendUpdates='all'
    ).execute()

    print(f"Event created: {created_event.get('htmlLink')}")
    return created_event
def book_appointment(kwargs):
    patient_name = kwargs.get("patient_name")
    doctor_specialty = kwargs.get("doctor_specialty")
    date_str = kwargs.get("date")
    parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Default time 9:00 AM Mountain Time
    mountain_tz = pytz.timezone("America/Denver")
    dt_mt = datetime.combine(parsed_date, time_1(9, 0))
    dt_mt = mountain_tz.localize(dt_mt)

    # Autodetect local timezone
    local_tz = get_localzone()
    dt_local = dt_mt.astimezone(local_tz)
    dt_local_end = dt_local + timedelta(hours=1)
    result = create_google_calendar_meeting(
        f"Meeting for {patient_name}",
        dt_local.isoformat(),
        dt_local_end.isoformat(),
        ["altondsouza02@gmail.com", "aldsouza@ualberta.ca"]
    )
    return {
        "status":200,
        "message": f"Event Created:{result}"
    }


function_execution_map = {
    "symptom_checker": symptom_checker,
    "medication_lookup": medication_lookup,
    "book_appointment": book_appointment
}


# Example prompt using the medical tools
# messages = [
#     {
#         "content": system_prompt,
#         "role": "system"
#     },
#     {
#         "content": (
#             "I have a headache and mild fever. What could be the possible conditions? "
#             "Also, lookup medication details for 'Ibuprofen'. "
#             "Please book an appointment for patient 'Alice Smith' with a neurologist on 2025-09-01."
#         ),
#         "role": "user"
#     }
# ]

# streamer = TextStreamer(tokenizer, skip_prompt=True)
# streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(mo)

# generation_kwargs = dict(inputs,streamer=streamer,
#         max_new_tokens=4096,
#         temperature=0.7,)
# thread = Thread(target=model.generate, kwargs=generation_kwargs,daemon=True)
# thread.start()
# for new_text in streamer:
#     print(new_text, end="")
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,streamer=streamer,
#         max_new_tokens=4096,
#         temperature=0.7,
#     )

class State(TypedDict):
    messages: List[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    task: str


graph_builder = StateGraph(State)

PLANNING_AGENT = "PLANNING_AGENT"


def planning(state: State):
    print("Coming up with Plan")
    # speak_text("I am processing your request and coming up with a plan")
    messages = state.get("messages", [])
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(inputs, streamer=streamer,
                             max_new_tokens=4096,
                             temperature=0.7, )
    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="")
        generated_text = generated_text + new_text
    generated_text = generated_text.replace("<｜end▁of▁sentence｜>","").replace("</think>","")
    # speak_text(generated_text)

    matches = regex.findall(pattern, generated_text, regex.VERBOSE)
    plan = state.get("plan", [])

    for i, (func_name, args_json) in enumerate(matches, 1):
        plan_entry = dict()
        plan_entry["function_name"] = func_name
        plan_entry["arguments"] = json.loads(args_json)
        plan.append(plan_entry)

    messages.append({"role": "assistant", "content": generated_text})

    return {"messages":messages, "plan": plan}


ROUTER = "ROUTER"


def router(state: State):
    plan = state.get("plan", [])
    if len(plan) > 0:
        return "execute_plan"
    return "respond"


def execute_plan(state: State):
    print("Executing")
    # speak_text("Executing Plan")
    plan = state.get("plan", [])
    for plan_entry in plan:
        plan_entry["status"] = dict()
        print(f"Executing {plan_entry['function_name']} with details {plan_entry['arguments']}")
        # speak_text("Approve execution of this plan by verifying details")
        print("Approve Execution?(y/n)")
        response = input()
        response = response.strip().lower()

        if response == "y":
            print("Approved.")
            if plan_entry["function_name"] in function_execution_map.keys():
                function = function_execution_map[plan_entry["function_name"]]
                result = function(plan_entry["arguments"])
                plan_entry["status"] = result
            else:
                print(f"Capability not implemented for {plan_entry['function_name']}")
            print("Done with task.")
            print("Proceeding with next.")

        elif response == "n":
            print("Not approved.")
        else:
            print("Invalid input, please enter 'y' or 'n'.")


    return {"plan": plan}


def respond(state: State):
    print(state.get("messages")[-1]["content"])
    return {"plan": state.get("plan")}


def summarize(state: State):
    plan = state.get("plan")
    messages = state.get("messages")
    summary_prompt = []
    summary_prompt.append({
        "role": "user","content": f"Summarize the results obtained from the following look at the status messages, the status=200 means the task executed successfully, comment on status and message and nothing else don't use json here just give me the details about the statuses from each entry :\n {json.dumps(plan,indent=2)}"
    })
    inputs = tokenizer.apply_chat_template(
        summary_prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(inputs, streamer=streamer,
                             max_new_tokens=4096,
                             temperature=0.7, )
    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="")
        generated_text = generated_text + new_text
    # speak_text(generated_text)
    messages.append({"role": "assistant", "content": generated_text})

    return {"messages":messages}


EXECUTE_PLAN = "EXECUTE_PLAN"
RESPOND = "RESPOND"
SUMMARIZE = "SUMMARIZE"
graph_builder.add_node(PLANNING_AGENT, planning)
graph_builder.add_node(EXECUTE_PLAN, execute_plan)
graph_builder.add_node(RESPOND, respond)
graph_builder.add_node(SUMMARIZE, summarize)

graph_builder.add_edge(START, PLANNING_AGENT)
graph_builder.add_conditional_edges(PLANNING_AGENT, router, {
    "execute_plan": EXECUTE_PLAN, "respond": RESPOND
})
graph_builder.add_edge(EXECUTE_PLAN, SUMMARIZE)
graph_builder.add_edge(SUMMARIZE, RESPOND)
graph_builder.add_edge(RESPOND, END)
compiled_graph = graph_builder.compile()
png_bytes = compiled_graph.get_graph().draw_mermaid_png()

# Save to file
with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph.png")

messages = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": (
            "I have a headache and mild fever. What could be the possible conditions? "
            "Also, lookup medication details for 'Ibuprofen'. "
            "Please book an appointment for patient 'Alice Smith' with a neurologist on 2025-08-25."
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
compiled_graph.invoke({"messages": messages})
# compiled_graph.invoke({"messages": different_user_prompt})

# while True:
#     command = input("Enter command (r = record, q = quit): ").strip().lower()
#
#     if command == 'q':
#         print("Quitting program...")
#         break
#     elif command == 'r':
#         with sr.Microphone() as source2:
#
#             # wait for a second to let the recognizer
#             # adjust the energy threshold based on
#             # the surrounding noise level
#             r.adjust_for_ambient_noise(source2, duration=1.0)
#             print("Recording started...")
#             # listens for the user's input
#             audio2 = r.listen(source2,phrase_time_limit=10,timeout=10)
#
#             # sphinx
#             MyText = r.recognize_vosk(audio2)
#             MyText = MyText.lower()
#             print(MyText)
#             messages = [
#                 {
#                     "content": system_prompt,
#                     "role": "system"
#                 },
#                 {
#                     "content": MyText,
#                     "role": "user"
#                 }
#             ]
#         print("Recording stopped.")
#         speak_text("Processing User Request")
#         compiled_graph.invoke({"messages":messages})
#     else:
#         print("Invalid command. Please enter 'r' or 'q'.")

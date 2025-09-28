import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, Response
from jinja2 import Template
import pickle
from datetime import datetime
from typing import TypedDict, Dict, List, Any

import pytz
from dotenv import load_dotenv
# from langchain_community.tools import TavilySearchResults  # Removed - using local medication database instead
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from regex import regex
from tzlocal import get_localzone
import pyttsx3
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_streaming_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Global variables for RKLLM
global_text = []
global_state = -1
is_blocking = False
lock = threading.Lock()

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT      = 0
RKLLMInputType.RKLLM_INPUT_TOKEN       = 1
RKLLMInputType.RKLLM_INPUT_EMBED       = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
        logger.info("RKLLM generation completed")
        # Flush any remaining TTS queue
        flush_tts_queue()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
        logger.error("RKLLM generation error")
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        text_chunk = result.contents.text.decode('utf-8')
        global_text += text_chunk
        # Log the generated text chunk
        print(text_chunk, end='', flush=True)
        logger.debug(f"Generated text chunk: {repr(text_chunk)}")
        
        # Add to TTS queue for sentence-based speaking
        add_to_tts_queue(text_chunk)
    return 0

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Health Agent Configuration
pattern = r'''
    \{                      # Opening brace of the function block
    \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
    \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
    (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
    \})                                # Closing brace of arguments
    \s*\}                             # Closing brace of the function block
    '''

# Load Jinja2 template from file
def load_chat_template():
    """Load the chat template from the Jinja2 file"""
    try:
        template_path = "chat_template.jinja"
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        return Template(template_content)
    except FileNotFoundError:
        print(f"Warning: Template file not found at {template_path}, using fallback template")
        # Fallback template
        fallback_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'REDACTED_SPECIAL_TOKEN' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'REDACTED_SPECIAL_TOKEN<|tool calls begin|><|tool call begin|>' + tool['type'] + '<|tool sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<|tool call end|>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<|tool call begin|>' + tool['type'] + '<|tool sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<|tool call end|>'}}{{'<|tool calls end|><|end of sentence|>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<|tool outputs end|>' + message['content'] + '<|end of sentence|>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'REDACTED_SPECIAL_TOKEN' + content + '<|end of sentence|>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<|tool outputs begin|><|tool output begin|>' + message['content'] + '<|tool output end|>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<|tool output begin|>' + message['content'] + '<|tool output end|>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<|tool outputs end|>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN<think>\n'}}{% endif %}"""
        return Template(fallback_template)
    except Exception as e:
        print(f"Error loading template: {e}, using fallback template")
        # Fallback template
        fallback_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'REDACTED_SPECIAL_TOKEN' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'REDACTED_SPECIAL_TOKEN<|tool calls begin|><|tool call begin|>' + tool['type'] + '<|tool sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<|tool call end|>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<|tool call begin|>' + tool['type'] + '<|tool sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<|tool call end|>'}}{{'<|tool calls end|><|end of sentence|>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<|tool outputs end|>' + message['content'] + '<|end of sentence|>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'REDACTED_SPECIAL_TOKEN' + content + '<|end of sentence|>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<|tool outputs begin|><|tool output begin|>' + message['content'] + '<|tool output end|>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<|tool output begin|>' + message['content'] + '<|tool output end|>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<|tool outputs end|>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN<think>\n'}}{% endif %}"""
        return Template(fallback_template)

# Load the chat template
chat_template = load_chat_template()

def apply_chat_template(messages, add_generation_prompt=True, bos_token="<|begin?of?sentence|>"):
    """Apply the Jinja2 chat template to messages"""
    try:
        return chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos_token
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        # Fallback to simple concatenation
        prompt = ""
        for message in messages:
            if message['role'] == 'system':
                prompt += f"System: {message['content']}\n"
            elif message['role'] == 'user':
                prompt += f"User: {message['content']}\n"
            elif message['role'] == 'assistant':
                prompt += f"Assistant: {message['content']}\n"
        return prompt

# Default medical tools configuration (can be overridden by client)
default_medical_tools = [
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
        "name": "check_heart_rate",
        "description": "Check if heart rate is within normal range (60-100 bpm).",
        "parameters": {
            "heart_rate": {
                "description": "Heart rate in beats per minute (bpm).",
                "type": "int",
                "default": 75
            }
        }
    },
    {
        "name": "check_temperature",
        "description": "Check if body temperature is within normal range (97.0-99.5 F).",
        "parameters": {
            "temperature": {
                "description": "Body temperature in Fahrenheit (F).",
                "type": "float",
                "default": 98.6
            }
        }
    }
]

# Default system prompt (can be overridden by client)
default_system_prompt = f"""
You are an intelligent AI assistant that uses available tools (functions) to help users achieve their medical-related goals. Your job is to understand the user's intent, identify missing information if needed, and then select and call the most appropriate function(s) to solve the task.

# Rules:
- ALWAYS use the tools provided to answer the user's request, unless explicitly told not to.
- Ask clarifying questions ONLY if the user's request is ambiguous or lacks required input parameters.
- If multiple tools are needed, use them in sequence.
- DO NOT make up data or assume values - request any missing input clearly.

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
{json.dumps(default_medical_tools, indent=2)}
"""

# Function execution map
function_execution_map = {
    "symptom_checker": "symptom_checker",
    "medication_lookup": "medication_lookup",
    "check_heart_rate": "check_heart_rate",
    "check_temperature": "check_temperature"
}

# State definition
class State(TypedDict):
    messages: List[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    task: str
    text_obj: Any
    use_tts: bool

# Function implementations
def symptom_checker(kwargs):
    print(f"Checking diseases for following symptoms on the web:")
    symptoms = kwargs.get("symptoms", [])
    print(symptoms)
    for i, arg in enumerate(symptoms):
        print(f"{i}. {arg}")
    results = TavilySearchResults()
    information = ""
    for result in results.invoke(f"What causes {''.join(symptoms)}"):
        information = information + result["content"] + "\n"
    return {
        "status": 200,
        "message": information
    }

def medication_lookup(kwargs):
    medication_name = kwargs.get("medication_name")
    print(f"Looking up information on {medication_name}....")
    
    # Mock medication database - replace with actual database in production
    medication_database = {
        "ibuprofen": {
            "name": "Ibuprofen",
            "type": "Non-steroidal anti-inflammatory drug (NSAID)",
            "uses": "Pain relief, fever reduction, inflammation",
            "dosage": "200-800mg every 4-6 hours as needed",
            "side_effects": "Stomach upset, heartburn, dizziness",
            "precautions": "Take with food, avoid alcohol, consult doctor if pregnant"
        },
        "aspirin": {
            "name": "Aspirin",
            "type": "Salicylate",
            "uses": "Pain relief, fever reduction, blood thinning",
            "dosage": "325-650mg every 4-6 hours as needed",
            "side_effects": "Stomach irritation, bleeding risk",
            "precautions": "Avoid in children with viral infections, consult doctor before surgery"
        },
        "nitroglycerin": {
            "name": "Nitroglycerin",
            "type": "Nitrate vasodilator",
            "uses": "Chest pain (angina), heart conditions",
            "dosage": "0.4mg sublingual as needed for chest pain",
            "side_effects": "Headache, dizziness, low blood pressure",
            "precautions": "Keep tablets in original container, avoid alcohol, seek immediate help for chest pain"
        },
        "acetaminophen": {
            "name": "Acetaminophen",
            "type": "Analgesic and antipyretic",
            "uses": "Pain relief, fever reduction",
            "dosage": "500-1000mg every 4-6 hours as needed",
            "side_effects": "Liver damage in high doses",
            "precautions": "Do not exceed 4000mg per day, avoid alcohol"
        }
    }
    
    # Convert medication name to lowercase for case-insensitive lookup
    med_key = medication_name.lower()
    
    if med_key in medication_database:
        med_info = medication_database[med_key]
        information = f"""
Medication Information for {med_info['name']}:

Type: {med_info['type']}
Uses: {med_info['uses']}
Typical Dosage: {med_info['dosage']}
Common Side Effects: {med_info['side_effects']}
Precautions: {med_info['precautions']}

Note: This is general information. Always consult with a healthcare provider for personalized medical advice.
"""
    else:
        information = f"""
Medication Information for {medication_name}:

This medication is not in our current database. Please consult with a healthcare provider or pharmacist for accurate information about {medication_name}.

General Safety Tips:
- Always read the label and follow dosage instructions
- Consult with a healthcare provider before taking new medications
- Be aware of potential drug interactions
- Store medications properly and keep out of reach of children
"""
    
    return {
        "status": 200,
        "message": information
    }



def check_heart_rate(kwargs):
    """Check if heart rate is within normal range"""
    heart_rate = kwargs.get("heart_rate")
    
    print(f"Checking heart rate: {heart_rate} bpm")
    
    # Normal ranges
    normal_heart_rate_min = 60
    normal_heart_rate_max = 100
    
    # Check heart rate
    heart_rate_status = "normal"
    if heart_rate < normal_heart_rate_min:
        heart_rate_status = "low (bradycardia)"
    elif heart_rate > normal_heart_rate_max:
        heart_rate_status = "high (tachycardia)"
    
    return {
        "status": 200,
        "message": {
            "heart_rate": {
                "value": heart_rate,
                "unit": "bpm",
                "status": heart_rate_status,
                "normal_range": f"{normal_heart_rate_min}-{normal_heart_rate_max} bpm"
            },
            "assessment": f"Heart rate is {heart_rate_status}"
        }
    }

def check_temperature(kwargs):
    """Check if body temperature is within normal range"""
    temperature = kwargs.get("temperature")
    
    print(f"Checking temperature: {temperature} F")
    
    # Normal ranges
    normal_temp_min = 97.0
    normal_temp_max = 99.5
    
    # Check temperature
    temp_status = "normal"
    if temperature < normal_temp_min:
        temp_status = "low (hypothermia)"
    elif temperature > normal_temp_max:
        temp_status = "high (fever)"
    
    return {
        "status": 200,
        "message": {
            "temperature": {
                "value": temperature,
                "unit": "F",
                "status": temp_status,
                "normal_range": f"{normal_temp_min}-{normal_temp_max} F"
            },
            "assessment": f"Temperature is {temp_status}"
        }
    }

# Build the LangGraph
def build_agent_graph():
    """Build and compile the LangGraph for the health agent"""
    
    # Build the graph
    graph_builder = StateGraph(State)
    
    # Node definitions
    PLANNING_AGENT = "PLANNING_AGENT"
    EXECUTE_PLAN = "EXECUTE_PLAN"
    RESPOND = "RESPOND"
    SUMMARIZE = "SUMMARIZE"
    
    def planning(state: State):
        print("Coming up with Plan")
        
        # Speak planning phase
        use_tts = state.get("use_tts", False)
        if use_tts:
            planning_text = "I'm analyzing your request and creating a plan to help you."
            speak_text(planning_text)
        
        messages = state.get("messages", [])
        
        # Apply chat template to messages
        full_prompt = apply_chat_template(messages, add_generation_prompt=True, bos_token="<|begin?of?sentence|>")
        logger.info(f"Planning prompt length: {len(full_prompt)}")
        logger.debug(f"Planning prompt: {repr(full_prompt)}")
        
        # Use RKLLM for generation
        if rkllm_model:
            try:
                # Reset global variables
                global global_text, global_state
                global_text = []
                global_state = -1
                
                # Run RKLLM model in a separate thread
                model_thread = threading.Thread(target=rkllm_model.run, args=("user", True, full_prompt))
                model_thread.start()
                
                # Wait for completion and collect generated text
                generated_text = ""
                model_thread_finished = False
                logger.info("Starting to collect generated text from planning...")
                while not model_thread_finished:
                    while len(global_text) > 0:
                        chunk = global_text.pop(0)
                        generated_text += chunk
                        logger.debug(f"Planning text chunk: {repr(chunk)}")
                        time.sleep(0.005)
                    
                    model_thread.join(timeout=0.005)
                    model_thread_finished = not model_thread.is_alive()
                
                logger.info(f"Planning completed. Total text length: {len(generated_text)}")
                logger.info(f"Planning generated text: {repr(generated_text)}")
                print(f"\n[PLANNING] Generated text: {generated_text}")
                
            except Exception as e:
                logger.error(f"Error in RKLLM planning: {e}")
                generated_text = "I apologize, but I encountered an error processing your request."
        else:
            generated_text = "I apologize, but the model is not available at the moment."
        
        generated_text = generated_text.replace("<|end?of?sentence|>", "").replace("</think>", "")

        matches = regex.findall(pattern, generated_text, regex.VERBOSE)
        plan = state.get("plan", [])

        for i, (func_name, args_json) in enumerate(matches, 1):
            plan_entry = dict()
            plan_entry["function_name"] = func_name
            plan_entry["arguments"] = json.loads(args_json)
            plan.append(plan_entry)

        messages.append({"role": "assistant", "content": generated_text})

        # Add generated text to TTS queue if TTS is enabled
        if use_tts:
            add_to_tts_queue(generated_text)

        # Speak the plan
        if use_tts:
            if plan:
                plan_text = f"I'll be executing {len(plan)} function(s) to help you."
                add_to_tts_queue(plan_text)
            else:
                no_plan_text = "I understand your request and will provide a direct response."
                add_to_tts_queue(no_plan_text)

        return {"messages": messages, "plan": plan}

    def router(state: State):
        plan = state.get("plan", [])
        if len(plan) > 0:
            return "execute_plan"
        return "respond"

    def execute_plan(state: State):
        print("Executing")
        
        # Speak execution phase
        use_tts = state.get("use_tts", False)
        if use_tts:
            execution_text = "Now I'm executing the planned functions to gather information for you."
            add_to_tts_queue(execution_text)
        
        plan = state.get("plan", [])
        for plan_entry in plan:
            plan_entry["status"] = dict()
            print(f"Executing {plan_entry['function_name']} with details {plan_entry['arguments']}")
            
            # Speak what function is being executed
            if use_tts:
                function_text = f"I'm now executing the {plan_entry['function_name']} function."
                add_to_tts_queue(function_text)
            
            if plan_entry["function_name"] in function_execution_map.keys():
                function_name = function_execution_map[plan_entry["function_name"]]
                if function_name == "symptom_checker":
                    result = symptom_checker(plan_entry["arguments"])
                elif function_name == "medication_lookup":
                    result = medication_lookup(plan_entry["arguments"])

                elif function_name == "check_heart_rate":
                    result = check_heart_rate(plan_entry["arguments"])
                elif function_name == "check_temperature":
                    result = check_temperature(plan_entry["arguments"])
                else:
                    result = {"status": 400, "message": "Function not implemented"}
                
                plan_entry["status"] = result
                print(f"Task completed successfully.")
            else:
                print(f"Capability not implemented for {plan_entry['function_name']}")
                plan_entry["status"] = {"status": 400, "message": "Function not implemented"}
            
            print("Proceeding with next.")

        # Speak completion of execution
        if use_tts:
            completion_text = "All functions have been executed successfully. Now I'll create a comprehensive summary for you."
            add_to_tts_queue(completion_text)

        return {"plan": plan}

    def respond(state: State):
        print(state.get("messages")[-1]["content"])
        
        # Speak response phase
        use_tts = state.get("use_tts", False)
        if use_tts:
            response_text = "Here's my response to your request."
            add_to_tts_queue(response_text)
        
        return {"plan": state.get("plan")}

    def summarize(state: State):
        plan = state.get("plan")
        messages = state.get("messages")
        
        # Speak summarization phase
        use_tts = state.get("use_tts", False)
        if use_tts:
            summary_text = "I'm now creating a comprehensive medical summary based on the gathered information."
            add_to_tts_queue(summary_text)
        
        # Create a summary of the executed functions and their results
        execution_summary = []
        for i, plan_entry in enumerate(plan, 1):
            function_name = plan_entry.get("function_name", "unknown")
            arguments = plan_entry.get("arguments", {})
            status_info = plan_entry.get("status", {})
            
            # Format arguments for readability
            args_summary = ", ".join([f"{k}: {v}" for k, v in arguments.items()])
            
            # Get status and message
            status_code = status_info.get("status", "unknown")
            message = status_info.get("message", "No message")
            
            # Create a readable summary for this function
            if status_code == 200:
                if function_name == "symptom_checker":
                    execution_summary.append(f"{i}. Symptom Analysis ({args_summary}) - SUCCESS: Analyzed symptoms and provided possible conditions")
                elif function_name == "medication_lookup":
                    execution_summary.append(f"{i}. Medication Lookup ({args_summary}) - SUCCESS: Retrieved detailed information about the medication")

                elif function_name == "check_heart_rate":
                    execution_summary.append(f"{i}. Heart Rate Check ({args_summary}) - SUCCESS: Heart rate assessment completed")
                elif function_name == "check_temperature":
                    execution_summary.append(f"{i}. Temperature Check ({args_summary}) - SUCCESS: Temperature assessment completed")
                else:
                    execution_summary.append(f"{i}. {function_name}({args_summary}) - SUCCESS: Task completed successfully")
            else:
                execution_summary.append(f"{i}. {function_name}({args_summary}) - FAILED (Status {status_code}): {message}")
        
        # Create summary messages for chat template
        summary_messages = [
            {
                "role": "system",
                "content": "You are a medical assistant that provides comprehensive summaries of medical function execution results. Use ONLY the information provided in the execution results. DO NOT hallucinate, make assumptions, or add information that is not explicitly stated. If information is missing or unclear, state that clearly rather than making up details."
            },
            {
                "role": "user", 
                "content": f"""Based on the following execution results, provide a comprehensive medical summary:

{chr(10).join(execution_summary)}

Please provide a structured summary that includes:

1. **Tasks Completed**: List what medical functions were successfully executed (based on the results above)
2. **Key Findings**: Highlight important medical information discovered from the actual results (symptoms, vitals, medication details, appointment links)
3. **Health Assessment**: Provide insights about the patient's condition based ONLY on the provided results
4. **Recommendations**: Suggest next steps or actions based on the actual findings (do not make up recommendations)
5. **Next Steps**: Suggest any follow-up actions based on the findings

Format your response as a clear, professional medical summary that a healthcare provider would understand. Focus on actionable insights and patient care recommendations based on the actual data provided."""
            }
        ]
        
        # Apply chat template to summary messages
        summary_prompt = apply_chat_template(summary_messages, add_generation_prompt=True, bos_token="<|begin?of?sentence|>")
        logger.info(f"Summarization prompt length: {len(summary_prompt)}")
        logger.debug(f"Summarization prompt: {repr(summary_prompt)}")
        
        # Use RKLLM for summarization
        if rkllm_model:
            try:
                # Reset global variables
                global global_text, global_state
                global_text = []
                global_state = -1
                
                # Run RKLLM model for summarization in a separate thread
                model_thread = threading.Thread(target=rkllm_model.run, args=("user", True, summary_prompt))
                model_thread.start()
                
                # Wait for completion and collect generated text
                generated_text = ""
                model_thread_finished = False
                logger.info("Starting to collect generated text from summarization...")
                while not model_thread_finished:
                    while len(global_text) > 0:
                        chunk = global_text.pop(0)
                        generated_text += chunk
                        logger.debug(f"Summarization text chunk: {repr(chunk)}")
                        time.sleep(0.005)
                    
                    model_thread.join(timeout=0.005)
                    model_thread_finished = not model_thread.is_alive()
                
                logger.info(f"Summarization completed. Total text length: {len(generated_text)}")
                logger.info(f"Summarization generated text: {repr(generated_text)}")
                print(f"\n[SUMMARIZATION] Generated text: {generated_text}")
                
            except Exception as e:
                logger.error(f"Error in RKLLM summarization: {e}")
                generated_text = "I apologize, but I encountered an error creating the summary."
        else:
            generated_text = "I apologize, but the model is not available for summarization."
        
        messages.append({"role": "assistant", "content": generated_text})

        # Add generated text to TTS queue if TTS is enabled
        if use_tts:
            add_to_tts_queue(generated_text)

        # Speak completion of summarization
        if use_tts:
            final_text = "I've completed the analysis and created a comprehensive summary for you."
            add_to_tts_queue(final_text)

        return {"messages": messages}

    # Add nodes to graph
    graph_builder.add_node(PLANNING_AGENT, planning)
    graph_builder.add_node(EXECUTE_PLAN, execute_plan)
    graph_builder.add_node(RESPOND, respond)
    graph_builder.add_node(SUMMARIZE, summarize)

    # Add edges
    graph_builder.add_edge(START, PLANNING_AGENT)
    graph_builder.add_conditional_edges(PLANNING_AGENT, router, {
        "execute_plan": EXECUTE_PLAN, "respond": RESPOND
    })
    graph_builder.add_edge(EXECUTE_PLAN, SUMMARIZE)
    graph_builder.add_edge(SUMMARIZE, RESPOND)
    graph_builder.add_edge(RESPOND, END)

    # Compile and return the graph
    compiled_graph = graph_builder.compile()
    
    # Save graph visualization
    png_bytes = compiled_graph.get_graph().draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Agent graph saved as agent_graph.png")
    
    return compiled_graph



# RKLLM Model class
class RKLLMModel:
    def __init__(self, model_path, platform='rk3588', max_context_len=2048, max_new_tokens=512):
        self.model_path = model_path
        self.platform = platform
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.handle = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the RKLLM model"""
        try:
            # Set up parameters based on platform
            extend_param = RKLLMExtendParam()
            extend_param.base_domain_id = 0
            extend_param.embed_flash = 1
            extend_param.n_batch = 1
            extend_param.use_cross_attn = 0
            
            # Platform-specific CPU configuration
            if self.platform == 'rk3588':
                extend_param.enabled_cpus_num = 4
                extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)  # CPUs 4-7
            elif self.platform == 'rk3576':
                extend_param.enabled_cpus_num = 4
                extend_param.enabled_cpus_mask = 0x0F  # CPUs 0-3
            else:
                # Default configuration
                extend_param.enabled_cpus_num = 4
                extend_param.enabled_cpus_mask = 0x0F
            
            param = RKLLMParam()
            param.model_path = self.model_path.encode('utf-8')
            param.max_context_len = self.max_context_len
            param.max_new_tokens = self.max_new_tokens
            param.top_k = 40
            param.n_keep = 0
            param.top_p = 0.8
            param.temperature = 0.7
            param.repeat_penalty = 1.1
            param.frequency_penalty = 0.0
            param.presence_penalty = 0.0
            param.mirostat = 0
            param.mirostat_tau = 5.0
            param.mirostat_eta = 0.1
            param.skip_special_token = True
            param.is_async = False
            param.img_start = "".encode('utf-8')
            param.img_end = "".encode('utf-8')
            param.img_content = "".encode('utf-8')
            param.extend_param = extend_param
            
            # Initialize model using the correct function name
            self.handle = RKLLM_Handle_t()
            self.rkllm_init = rkllm_lib.rkllm_init
            self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
            self.rkllm_init.restype = ctypes.c_int
            result = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), callback)
            
            if result != 0:
                raise Exception(f"Failed to initialize RKLLM model with error code {result}")
            
            # Set up function signatures for chat template
            self.set_chat_template = rkllm_lib.rkllm_set_chat_template
            self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
            self.set_chat_template.restype = ctypes.c_int
            
            # Set up run function
            self.rkllm_run = rkllm_lib.rkllm_run
            self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
            self.rkllm_run.restype = ctypes.c_int
            
            # Set up destroy function
            self.rkllm_destroy = rkllm_lib.rkllm_destroy
            self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
            self.rkllm_destroy.restype = ctypes.c_int
            
            # Set up infer parameters
            self.rkllm_infer_params = RKLLMInferParam()
            ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
            self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            self.rkllm_infer_params.lora_params = None
            self.rkllm_infer_params.keep_history = 0
            
            logger.info(f"RKLLM model initialized successfully from {self.model_path} for platform {self.platform}")
                
        except Exception as e:
            logger.error(f"Error initializing RKLLM model: {str(e)}")
            raise
    
    def run(self, role="user", enable_thinking=True, prompt=""):
        """Run the RKLLM model with the given prompt"""
        global global_text, global_state
        
        try:
            if not self.handle:
                raise Exception("Model not initialized")
            
            # Reset global variables
            global_text = []
            global_state = -1
            
            # Prepare the input
            rkllm_input = RKLLMInput()
            rkllm_input.role = role.encode('utf-8')
            rkllm_input.enable_thinking = enable_thinking
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = prompt.encode('utf-8')
            
            # Call the model using the correct function
            result = self.rkllm_run(
                self.handle,
                ctypes.byref(rkllm_input),
                ctypes.byref(self.rkllm_infer_params),
                userdata
            )
            
            if result != 0:
                raise Exception(f"rkllm_run failed with error code {result}")
            
            logger.info("RKLLM model run completed successfully")
            
        except Exception as e:
            logger.error(f"Error in RKLLM model run: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up the model resources"""
        if self.handle:
            self.rkllm_destroy(self.handle)
            self.handle = None
            logger.info("RKLLM model cleaned up")

# Initialize RKLLM model (will be initialized after args are parsed)
rkllm_model = None

# Initialize TTS engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume level
    logger.info("TTS engine initialized")
except Exception as e:
    logger.warning(f"Could not initialize TTS engine: {e}")
    tts_engine = None

# TTS queue and state management
tts_queue = []
tts_lock = threading.Lock()
tts_thread = None
tts_running = False
current_sentence = ""
word_count = 0
SENTENCE_ENDINGS = ['.', '!', '?', ':', ';']

def add_to_tts_queue(text):
    """Add text to TTS queue for sentence-based speaking"""
    global current_sentence, word_count
    
    if not tts_engine:
        return
    
    with tts_lock:
        # Split text into words
        words = text.split()
        
        for word in words:
            current_sentence += word + " "
            word_count += 1
            
            # Check if we should speak (every 10 words or sentence end)
            should_speak = False
            
            # Check for sentence endings
            if any(ending in word for ending in SENTENCE_ENDINGS):
                should_speak = True
            # Check if we've reached 10 words
            elif word_count >= 10:
                should_speak = True
            
            if should_speak and current_sentence.strip():
                # Add to queue and reset counters
                sentence_to_speak = current_sentence.strip()
                tts_queue.append(sentence_to_speak)
                current_sentence = ""
                word_count = 0
                
                # Start TTS thread if not running
                start_tts_thread()

def flush_tts_queue():
    """Flush any remaining text in the current sentence"""
    global current_sentence, word_count
    
    if current_sentence.strip():
        with tts_lock:
            tts_queue.append(current_sentence.strip())
            current_sentence = ""
            word_count = 0
            start_tts_thread()

def start_tts_thread():
    """Start the TTS processing thread if not already running"""
    global tts_thread, tts_running
    
    if not tts_running:
        tts_running = True
        tts_thread = threading.Thread(target=process_tts_queue, daemon=True)
        tts_thread.start()
        logger.info("TTS processing thread started")

def process_tts_queue():
    """Process the TTS queue in a separate thread"""
    global tts_running
    
    while tts_running:
        try:
            # Get next sentence from queue
            sentence = None
            with tts_lock:
                if tts_queue:
                    sentence = tts_queue.pop(0)
                else:
                    tts_running = False
                    break
            
            if sentence and tts_engine:
                try:
                    logger.info(f"TTS speaking: {repr(sentence)}")
                    tts_engine.say(sentence)
                    tts_engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS error: {e}")
            
            # Small delay to prevent overwhelming the TTS engine
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in TTS processing thread: {e}")
            break
    
    logger.info("TTS processing thread stopped")

def speak_text(text):
    """Legacy function - now uses sentence-based queuing"""
    add_to_tts_queue(text)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    client_ip = request.remote_addr
    logger.info(f"Health check request from {client_ip}")
    return jsonify({
        'status': 'healthy',
        'agent_loaded': True,
        'rkllm_loaded': rkllm_model is not None,
        'tts_loaded': tts_engine is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for health agent"""
    global is_blocking
    
    if is_blocking:
        return jsonify({'error': 'Server is busy processing another request'}), 503
    
    try:
        lock.acquire()
        is_blocking = True
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        use_tts = data.get('use_tts', False)
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Add system prompt if not present
        if not any(msg.get('role') == 'system' for msg in messages):
            messages.insert(0, {
                "content": default_system_prompt,
                "role": "system"
            })
        
        logger.info(f"Processing chat request with {len(messages)} messages, TTS: {use_tts}")
        
        if stream:
            return Response(stream_agent_response(messages, use_tts), content_type='text/plain')
        else:
            # Non-streaming response
            result = agent_graph.invoke({"messages": messages, "use_tts": use_tts})
            
            # Handle TTS if requested
            if use_tts:
                final_messages = result.get("messages", [])
                if final_messages:
                    final_response = final_messages[-1].get("content", "")
                    if final_response and final_messages[-1].get("role") == "assistant":
                        # Add final response to TTS queue
                        add_to_tts_queue(final_response)
                        # Flush any remaining text
                        flush_tts_queue()
            
            return jsonify({
                "messages": result.get("messages", []),
                "plan": result.get("plan", []),
                "status": "success"
            })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        lock.release()
        is_blocking = False

def stream_agent_response(messages, use_tts=False):
    """Stream the agent response"""
    try:
        logger.info("Starting streaming agent response")
        
        # Run the agent graph
        result = agent_graph.invoke({"messages": messages, "use_tts": use_tts})
        
        # Get the final response
        final_messages = result.get("messages", [])
        if final_messages:
            final_response = final_messages[-1].get("content", "")
            
            # Stream the response in chunks
            chunk_size = 50
            for i in range(0, len(final_response), chunk_size):
                chunk = final_response[i:i + chunk_size]
                yield f"data: {json.dumps({'content': chunk})}\n\n"
                time.sleep(0.01)
            
            # Handle TTS if requested
            if use_tts and final_messages[-1].get("role") == "assistant":
                # Add final response to TTS queue
                add_to_tts_queue(final_response)
                # Flush any remaining text
                flush_tts_queue()
        
        yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming agent response: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/generate', methods=['POST'])
def generate():
    """RKLLM generation endpoint"""
    global global_text, global_state, is_blocking
    
    if is_blocking:
        return jsonify({'error': 'Server is busy processing another request'}), 503
    
    try:
        lock.acquire()
        is_blocking = True
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        messages = data.get('messages', [])
        prompt = data.get('prompt', '')
        role = data.get('role', 'user')
        enable_thinking = data.get('enable_thinking', True)
        stream = data.get('stream', False)
        add_generation_prompt = data.get('add_generation_prompt', True)
        bos_token = data.get('bos_token', '<|begin?of?sentence|>')
        
        # Use messages if provided, otherwise use prompt
        if messages:
            # Apply chat template to messages
            full_prompt = apply_chat_template(messages, add_generation_prompt, bos_token)
            logger.info(f"Generate endpoint prompt length: {len(full_prompt)}")
            logger.debug(f"Generate endpoint prompt: {repr(full_prompt)}")
        elif prompt:
            full_prompt = prompt
        else:
            return jsonify({'error': 'No prompt or messages provided'}), 400
        
        if not rkllm_model:
            return jsonify({'error': 'RKLLM model not available'}), 503
        
        logger.info(f"Processing generation request: role={role}, thinking={enable_thinking}, stream={stream}")
        
        # Reset global variables
        global_text = []
        global_state = -1
        
        if stream:
            return Response(stream_generate(full_prompt, role, enable_thinking), content_type='text/plain')
        else:
            # Non-streaming generation
            model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, full_prompt))
            model_thread.start()
            
            # Wait for completion
            rkllm_output = ""
            model_thread_finished = False
            logger.info("Starting to collect generated text from /generate endpoint...")
            while not model_thread_finished:
                while len(global_text) > 0:
                    chunk = global_text.pop(0)
                    rkllm_output += chunk
                    logger.debug(f"Generate endpoint text chunk: {repr(chunk)}")
                    time.sleep(0.005)
                
                model_thread.join(timeout=0.005)
                model_thread_finished = not model_thread.is_alive()
            
            logger.info(f"Generate endpoint completed. Total text length: {len(rkllm_output)}")
            logger.info(f"Generate endpoint output: {repr(rkllm_output)}")
            
            return jsonify({
                "text": rkllm_output,
                "status": "success"
            })
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        lock.release()
        is_blocking = False

def stream_generate(full_prompt, role="user", enable_thinking=True, stream_granularity="chunk"):
    """Stream generation function"""
    global global_text, global_state
    
    try:
        logger.info(f"Starting model thread for streaming")
        model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, full_prompt))
        model_thread.start()
        
        model_thread_finished = False
        chunk_count = 0
        logger.info("Starting to collect streaming text chunks...")
        while not model_thread_finished:
            while len(global_text) > 0:
                chunk = global_text.pop(0)
                chunk_count += 1
                
                logger.debug(f"Streaming chunk {chunk_count}: {repr(chunk)}")
                if chunk_count % 10 == 0:  # Log every 10 chunks
                    logger.info(f"Streaming chunk {chunk_count}, content: {repr(chunk[:50])}...")
                
                # Handle different streaming granularities
                if stream_granularity == "token":
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    time.sleep(0.001)
                elif stream_granularity == "character":
                    for char in chunk:
                        yield f"data: {json.dumps({'content': char})}\n\n"
                        time.sleep(0.001)
                elif stream_granularity == "word":
                    import re
                    parts = re.split(r'(\s+)', chunk)
                    for part in parts:
                        if part:
                            yield f"data: {json.dumps({'content': part})}\n\n"
                            time.sleep(0.001)
                else:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    time.sleep(0.001)
            
            model_thread.join(timeout=0.005)
            model_thread_finished = not model_thread.is_alive()
        
        logger.info(f"Streaming completed, total chunks: {chunk_count}")
        yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
    
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agent Streaming Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--model-path', default='/path/to/your/rkllm/model', help='Path to RKLLM model')
    parser.add_argument('--target-platform', type=str, default='rk3588', choices=['rk3588', 'rk3576'], 
                       help='Target platform: rk3588 or rk3576')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Validate platform
    if args.target_platform not in ["rk3588", "rk3576"]:
        print("Error: Please specify the correct target platform: rk3588 or rk3576.")
        sys.exit(1)
    
    # Fix frequency for the target platform
    try:
        command = f"sudo bash fix_freq_{args.target_platform}.sh"
        subprocess.run(command, shell=True, check=False)
        logger.info(f"Frequency fix script executed for {args.target_platform}")
    except Exception as e:
        logger.warning(f"Could not execute frequency fix script: {e}")
    
    if args.debug:
        app.debug = True
    
    # Initialize RKLLM model
    try:
        rkllm_model = RKLLMModel(args.model_path, platform=args.target_platform)
        logger.info(f"RKLLM model initialized for platform: {args.target_platform}")
    except Exception as e:
        logger.warning(f"Could not initialize RKLLM model: {e}")
        rkllm_model = None
    
    # Initialize the agent graph
    agent_graph = build_agent_graph()
    
    logger.info(f"Starting Agent Streaming Server on {args.host}:{args.port}")
    logger.info(f"Target platform: {args.target_platform}")
    logger.info(f"Health agent initialized with {len(default_medical_tools)} tools")
    logger.info(f"TTS engine: {'Available' if tts_engine else 'Not available'}")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        # Clean up TTS system
        tts_running = False
        flush_tts_queue()
        
        if rkllm_model:
            rkllm_model.cleanup()
        if tts_engine:
            tts_engine.stop()
        logger.info("Server shutdown complete")

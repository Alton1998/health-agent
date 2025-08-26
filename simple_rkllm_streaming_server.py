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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rkllm_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    client_ip = request.remote_addr
    logger.info(f"Health check request from {client_ip}")
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

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

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output
global_text = []
global_state = -1

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        global_text += result.contents.text.decode('utf-8')
    return 0

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

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
        fallback_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'REDACTED_SPECIAL_TOKEN' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'REDACTED_SPECIAL_TOKEN<｜tool calls begin｜><｜tool call begin｜>' + tool['type'] + '<｜tool sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool call end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool call begin｜>' + tool['type'] + '<｜tool sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool call end｜>'}}{{'<｜tool calls end｜><｜end of sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool outputs end｜>' + message['content'] + '<｜end of sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'REDACTED_SPECIAL_TOKEN' + content + '<｜end of sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool outputs begin｜><｜tool output begin｜>' + message['content'] + '<｜tool output end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool output begin｜>' + message['content'] + '<｜tool output end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool outputs end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN<think>\n'}}{% endif %}"""
        return Template(fallback_template)
    except Exception as e:
        print(f"Error loading template: {e}, using fallback template")
        # Fallback template
        fallback_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'REDACTED_SPECIAL_TOKEN' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'REDACTED_SPECIAL_TOKEN<｜tool calls begin｜><｜tool call begin｜>' + tool['type'] + '<｜tool sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool call end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool call begin｜>' + tool['type'] + '<｜tool sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool call end｜>'}}{{'<｜tool calls end｜><｜end of sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool outputs end｜>' + message['content'] + '<｜end of sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'REDACTED_SPECIAL_TOKEN' + content + '<｜end of sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool outputs begin｜><｜tool output begin｜>' + message['content'] + '<｜tool output end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool output begin｜>' + message['content'] + '<｜tool output end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool outputs end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN<think>\n'}}{% endif %}"""
        return Template(fallback_template)

# Load the chat template
chat_template = load_chat_template()

def apply_chat_template(messages, add_generation_prompt=True, bos_token="<｜begin▁of▁sentence｜>"):
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

# Define the RKLLM class
class RKLLM(object):
    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = 4096
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4
        rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int
        
        self.rkllm_abort = rkllm_lib.rkllm_abort

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
        
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def run(self, *param):
        role, enable_thinking, prompt = param
        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode('utf-8') if role is not None else "user".encode('utf-8')
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking if enable_thinking is not None else False)
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return
    
    def abort(self):
        return self.rkllm_abort(self.handle)
    
    def release(self):
        self.rkllm_destroy(self.handle)

# Global RKLLM model instance
rkllm_model = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": rkllm_model is not None,
        "timestamp": time.time()
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text endpoint"""
    global global_text, global_state, is_blocking
    
    if is_blocking or global_state == 0:
        return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
    
    lock.acquire()
    try:
        is_blocking = True
        
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt required'}), 400
        
        prompt = data['prompt']
        role = data.get('role', 'user')
        enable_thinking = data.get('enable_thinking', True)
        stream = data.get('stream', False)
        
        # Reset global variables
        global_text = []
        global_state = -1
        
        if stream:
            return Response(stream_generate(prompt, role, enable_thinking), content_type='text/plain')
        else:
            # Non-streaming generation
            model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, prompt))
            model_thread.start()
            
            # Wait for completion
            rkllm_output = ""
            model_thread_finished = False
            while not model_thread_finished:
                while len(global_text) > 0:
                    rkllm_output += global_text.pop(0)
                    time.sleep(0.005)
                
                model_thread.join(timeout=0.005)
                model_thread_finished = not model_thread.is_alive()
            
            return jsonify({
                "text": rkllm_output,
                "status": "success"
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        lock.release()
        is_blocking = False

def stream_generate(prompt, request_id=None, role="user", enable_thinking=True, stream_granularity="chunk"):
    """
    Stream generation function
    stream_granularity: "chunk" (default), "word", "character", or "token" (raw model output)
    """
    """Stream generation function"""
    global global_text, global_state
    
    try:
        if request_id:
            logger.info(f"[{request_id}] Starting model thread for streaming")
        model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, prompt))
        model_thread.start()
        
        model_thread_finished = False
        chunk_count = 0
        while not model_thread_finished:
            while len(global_text) > 0:
                chunk = global_text.pop(0)
                chunk_count += 1
                if request_id and chunk_count % 10 == 0:  # Log every 10 chunks
                    logger.info(f"[{request_id}] Streaming chunk {chunk_count}, content: {repr(chunk[:50])}...")
                
                # Handle different streaming granularities
                if stream_granularity == "token":
                    # Stream each token exactly as received from the model (raw output)
                    if request_id and chunk_count <= 5:  # Log first 5 tokens for debugging
                        logger.info(f"[{request_id}] Token {chunk_count}: {repr(chunk)}")
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    time.sleep(0.001)
                elif stream_granularity == "character":
                    # Stream each character exactly as received
                    for char in chunk:
                        yield f"data: {json.dumps({'content': char})}\n\n"
                        time.sleep(0.001)
                elif stream_granularity == "word":
                    # Stream each word exactly as received, preserving original spacing
                    # Split on whitespace but preserve the original text structure
                    import re
                    # Split on whitespace while keeping the original spacing
                    parts = re.split(r'(\s+)', chunk)
                    for part in parts:
                        if part:  # Skip empty parts
                            yield f"data: {json.dumps({'content': part})}\n\n"
                            time.sleep(0.001)
                else:
                    # Stream each chunk (default) exactly as received
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    time.sleep(0.001)
            
            model_thread.join(timeout=0.005)
            model_thread_finished = not model_thread.is_alive()
        
        if request_id:
            logger.info(f"[{request_id}] Streaming completed, total chunks: {chunk_count}")
        yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
    
    except Exception as e:
        if request_id:
            logger.error(f"[{request_id}] Streaming error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with Jinja2 template support"""
    global global_text, global_state, is_blocking
    
    # Log request details
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    client_ip = request.remote_addr
    logger.info(f"[{request_id}] Chat request received from {client_ip}")
    
    if is_blocking or global_state == 0:
        logger.warning(f"[{request_id}] Server busy, rejecting request")
        return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
    
    lock.acquire()
    try:
        is_blocking = True
        logger.info(f"[{request_id}] Request processing started")
        
        data = request.json
        if not data or 'messages' not in data:
            logger.error(f"[{request_id}] Invalid request: messages required")
            return jsonify({'error': 'Messages required'}), 400
        
        messages = data['messages']
        stream = data.get('stream', False)
        add_generation_prompt = data.get('add_generation_prompt', True)
        bos_token = data.get('bos_token', '<｜begin▁of▁sentence｜>')
        
        # Apply Jinja2 chat template
        prompt = apply_chat_template(messages, add_generation_prompt, bos_token)
        logger.info(f"[{request_id}] Template applied, prompt length: {len(prompt)} characters")
        
        # Reset global variables
        global_text = []
        global_state = -1
        
        if stream:
            logger.info(f"[{request_id}] Starting streaming generation")
            stream_granularity = data.get('stream_granularity', 'chunk')  # chunk, word, or character
            return Response(stream_generate(prompt, request_id, stream_granularity=stream_granularity), content_type='text/plain')
        else:
            logger.info(f"[{request_id}] Starting non-streaming generation")
            # Non-streaming generation
            model_thread = threading.Thread(target=rkllm_model.run, args=("user", True, prompt))
            model_thread.start()
            
            # Wait for completion
            rkllm_output = ""
            model_thread_finished = False
            while not model_thread_finished:
                while len(global_text) > 0:
                    rkllm_output += global_text.pop(0)
                    time.sleep(0.005)
                
                model_thread.join(timeout=0.005)
                model_thread_finished = not model_thread.is_alive()
            
            logger.info(f"[{request_id}] Non-streaming generation completed, output length: {len(rkllm_output)} characters")
            return jsonify({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": rkllm_output
                    }
                }]
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        lock.release()
        is_blocking = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True, help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path:
        if not os.path.exists(args.lora_model_path):
            print("Error: Please provide the correct lora_model path, and ensure it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    if args.prompt_cache_path:
        if not os.path.exists(args.prompt_cache_path):
            print("Error: Please provide the correct prompt_cache_file path, and ensure it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize RKLLM model
    logger.info("========= Initializing RKLLM model =========")
    print("=========init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, args.lora_model_path, args.prompt_cache_path)
    logger.info("RKLLM Model has been initialized successfully!")
    print("RKLLM Model has been initialized successfully！")
    print("==============================")
    sys.stdout.flush()
    
    logger.info("Simple RKLLM Streaming Server with Jinja2 Template starting...")
    print("Simple RKLLM Streaming Server with Jinja2 Template starting...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /generate - Generate text (with streaming support)")
    print("  POST /chat - Chat endpoint with Jinja2 template (with streaming support)")
    print(f"Server will run on port {args.port}")
    logger.info(f"Server starting on port {args.port}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")

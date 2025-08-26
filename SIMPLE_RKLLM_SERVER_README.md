# Simple RKLLM Streaming Server

A lightweight Flask server that provides RKLLM model inference with streaming capabilities, without any function calling or tools.

## Features

- **RKLLM Integration**: Uses RKLLM for optimized model inference on RK3588/RK3576 platforms
- **Streaming Support**: Real-time streaming responses for both generate and chat endpoints
- **Simple API**: Clean, minimal API without function calling complexity
- **Threading**: Multi-threaded server with proper locking for concurrent requests
- **Health Monitoring**: Built-in health check endpoint
- **Error Handling**: Comprehensive error handling and validation

## Prerequisites

1. **RK3588 or RK3576 platform** - The server is designed for these specific platforms
2. **RKLLM runtime library** - You need `lib/librkllmrt.so` in your project
3. **Converted RKLLM model** - Your model must be converted to RKLLM format
4. **Python dependencies** - Install the required packages

## Installation

Install the required dependencies:

```bash
pip install flask requests
```

## Running the Server

The server requires several command-line arguments:

```bash
python simple_rkllm_streaming_server.py \
    --rkllm_model_path /path/to/your/converted/model \
    --target_platform rk3588 \
    --lora_model_path /path/to/lora/model \
    --prompt_cache_path /path/to/prompt_cache \
    --port 5000
```

### Required Arguments

- `--rkllm_model_path`: Absolute path to your converted RKLLM model
- `--target_platform`: Target platform (`rk3588` or `rk3576`)

### Optional Arguments

- `--lora_model_path`: Path to LoRA adapter model (if using LoRA)
- `--prompt_cache_path`: Path to prompt cache file (if using prompt caching)
- `--port`: Port to run the server on (default: 5000)

## API Endpoints

### 1. Health Check

**GET** `/health`

Check server health and model status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": 1234567890.123
}
```

### 2. Generate Text

**POST** `/generate`

Generate text from a prompt.

**Request Body:**
```json
{
    "prompt": "Your input prompt here",
    "role": "user",
    "enable_thinking": true,
    "stream": false
}
```

**Parameters:**
- `prompt` (required): The input text prompt
- `role` (optional): Role for the prompt (default: "user")
- `enable_thinking` (optional): Enable thinking mode (default: true)
- `stream` (optional): Enable streaming response (default: false)

**Non-streaming Response:**
```json
{
    "text": "Generated text response",
    "status": "success"
}
```

**Streaming Response:**
Server-sent events with format:
```
data: {"content": "chunk1"}
data: {"content": "chunk2"}
data: {"content": "[DONE]"}
```

### 3. Chat

**POST** `/chat`

Chat endpoint for conversation-style interactions.

**Request Body:**
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": false
}
```

**Parameters:**
- `messages` (required): Array of message objects with role and content
- `stream` (optional): Enable streaming response (default: false)

**Non-streaming Response:**
```json
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Generated response"
            }
        }
    ]
}
```

**Streaming Response:**
Server-sent events with format:
```
data: {"content": "chunk1"}
data: {"content": "chunk2"}
data: {"content": "[DONE]"}
```

## Testing

Use the provided test client to verify the server functionality:

```bash
python test_simple_rkllm_server.py
```

The test client will:
- Check server health
- Test non-streaming generation
- Test streaming generation
- Test non-streaming chat
- Test streaming chat
- Test error handling

## Example Usage

### Using curl

**Health check:**
```bash
curl http://localhost:5000/health
```

**Non-streaming generation:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "stream": false}'
```

**Streaming generation:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "stream": true}' \
  --no-buffer
```

**Chat with streaming:**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is AI?"}], "stream": true}' \
  --no-buffer
```

### Using Python requests

```python
import requests
import json

# Non-streaming generation
response = requests.post('http://localhost:5000/generate', json={
    'prompt': 'Hello, how are you?',
    'stream': False
})
print(response.json()['text'])

# Streaming generation
response = requests.post('http://localhost:5000/generate', json={
    'prompt': 'Tell me a story',
    'stream': True
}, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            data_json = json.loads(data_str)
            content = data_json.get('content', '')
            if content == '[DONE]':
                break
            print(content, end='', flush=True)
```

## Architecture

The server uses:

- **Flask**: Web framework for HTTP endpoints
- **ctypes**: Python foreign function library to interface with RKLLM C++ runtime
- **Threading**: For concurrent request handling and non-blocking model inference
- **Server-Sent Events**: For streaming responses
- **Global state management**: For tracking model inference status

## Error Handling

The server handles various error conditions:

- **503 Service Unavailable**: When the server is busy processing another request
- **400 Bad Request**: When required parameters are missing
- **500 Internal Server Error**: For unexpected errors during model inference

## Performance Considerations

- **Resource limits**: Sets file descriptor limits to 102400
- **CPU optimization**: Configures CPU masks for optimal performance
- **Memory management**: Proper cleanup of model resources
- **Concurrent requests**: Threading with locks to prevent conflicts

## Troubleshooting

### Common Issues

1. **Model path not found**: Ensure the RKLLM model path is correct and absolute
2. **Library not found**: Verify `lib/librkllmrt.so` exists and is accessible
3. **Platform mismatch**: Ensure target platform is either `rk3588` or `rk3576`
4. **Permission denied**: Run frequency fix script with sudo if needed

### Debug Mode

For debugging, you can modify the Flask app to run in debug mode:

```python
app.run(host='0.0.0.0', port=args.port, debug=True, threaded=True)
```

## Security Considerations

- The server runs on `0.0.0.0` by default, making it accessible from any network interface
- Consider implementing authentication and rate limiting for production use
- Validate all input data to prevent injection attacks

## Future Enhancements

- Add authentication and authorization
- Implement rate limiting
- Add metrics and monitoring
- Support for model parameter tuning via API
- Add WebSocket support for real-time communication

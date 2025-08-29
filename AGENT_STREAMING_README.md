# Agent Streaming Server

A Flask-based streaming server that combines RKLLM streaming functionality with a health agent workflow, providing a REST API for medical assistance without voice/TTS components.

## Features

- **Health Agent Workflow**: LangGraph-based agent with medical tools
- **Streaming Responses**: Real-time streaming of agent responses
- **RKLLM Integration**: Optional RKLLM model support for additional generation capabilities
- **Medical Tools**: Symptom checking, medication lookup, appointment booking, vital signs monitoring
- **REST API**: Simple HTTP endpoints for easy integration
- **Threading**: Non-blocking request handling with proper resource management
- **Client-Side Configuration**: System prompts and medical tools can be customized per client
- **Voice Recognition**: Speech-to-text input using Vosk (offline speech recognition)
- **Text-to-Speech**: TTS output using pyttsx3 with configurable settings throughout the agent workflow

## Medical Tools Available

1. **symptom_checker**: Analyze symptoms and provide possible conditions
2. **medication_lookup**: Look up details about medications
3. **book_appointment**: Schedule medical appointments with Google Calendar integration
4. **check_heart_rate**: Check if heart rate is within normal range (60-100 bpm)
5. **check_temperature**: Check if body temperature is within normal range (97.0-99.5°F)

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Dependencies

- Flask
- LangGraph
- LangChain
- Google API Client (for calendar integration)
- Tavily Search (for web searches)
- Regex
- PyTZ
- TZLocal
- SpeechRecognition (for voice input)
- pyttsx3 (for text-to-speech)
- PyAudio (for microphone access)
- Vosk (for offline speech recognition)

### Optional Dependencies

- RKLLM library (`lib/librkllmrt.so`) for additional model support
- Platform-specific frequency fix scripts (`fix_freq_rk3588.sh`, `fix_freq_rk3576.sh`)
- Chat template file (`chat_template.jinja`) for custom message formatting

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
```

### Google Calendar Setup

1. Create a Google Cloud Project
2. Enable Google Calendar API
3. Create OAuth 2.0 credentials
4. Download `credentials.json` to your project directory

### Vosk Speech Recognition Setup

1. Download a Vosk model from https://alphacephei.com/vosk/models
2. Extract the model to your project directory
3. The default model path is `vosk-model-small-en-us-0.15`
4. You can use other models by modifying the `vosk_model_path` in the client

### Chat Template Setup

1. Create a `chat_template.jinja` file in your project directory for custom message formatting
2. If no template file is found, the system will use a built-in fallback template
3. The template supports system, user, assistant, and tool message roles
4. Template variables include `messages`, `add_generation_prompt`, and `bos_token`

## Usage

### Starting the Server

```bash
# Basic usage
python agent_streaming_server.py

# With custom host and port
python agent_streaming_server.py --host 0.0.0.0 --port 5000

# With RKLLM model path and platform
python agent_streaming_server.py --model-path /path/to/rkllm/model --target-platform rk3588

# With different platform
python agent_streaming_server.py --model-path /path/to/rkllm/model --target-platform rk3576

# Debug mode
python agent_streaming_server.py --debug
```

### Using the Client

```bash
# Interactive mode
python agent_client.py

# Run examples
python agent_client.py example
```

### Client Configuration

The client allows you to customize the system prompt and medical tools:

#### Voice and TTS Features

The client supports voice input and TTS output:

```python
from agent_client import AgentClient

client = AgentClient()

# Voice input methods (using Vosk offline recognition)
voice_text = client.get_voice_input()  # One-time voice input
client.start_voice_listening()         # Start background listening
voice_text = client.get_queued_voice_input()  # Get from background queue
client.stop_voice_listening()          # Stop background listening

# Set custom Vosk model
client.set_vosk_model_path("path/to/your/vosk/model")
model_path = client.get_vosk_model_path()

# Send message with TTS (speaks throughout all agent phases)
result = client.send_message("Hello", use_tts=True)

# Interactive mode commands:
# 'tts' - Toggle TTS mode
# 'voice' - Toggle voice input mode  
# 'voice_start' - Start background voice listening
# 'voice_stop' - Stop background voice listening
```

```python
from agent_client import AgentClient

client = AgentClient()

# Set custom system prompt
client.set_system_prompt("You are a specialized cardiology assistant...")

# Set custom medical tools
custom_tools = [
    {
        "name": "check_heart_rate",
        "description": "Check heart rate",
        "parameters": {...}
    }
]
client.set_medical_tools(custom_tools)

# Get current configuration
tools = client.get_medical_tools()
prompt = client.get_system_prompt()
```

## API Endpoints

### Health Check

**GET** `/health`

Check server status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "agent_loaded": true,
  "rkllm_loaded": false,
  "tts_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Chat with Agent

**POST** `/chat`

Send messages to the health agent and get responses.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "I have a headache and fever. What could be wrong?"
    }
  ],
  "stream": false,
  "use_tts": false
}
```

**Response (Non-streaming):**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt..."
    },
    {
      "role": "user",
      "content": "I have a headache and fever. What could be wrong?"
    },
    {
      "role": "assistant",
      "content": "Based on your symptoms, I'll analyze them and provide information..."
    }
  ],
  "plan": [
    {
      "function_name": "symptom_checker",
      "arguments": {
        "symptoms": ["headache", "fever"]
      },
      "status": {
        "status": 200,
        "message": "Analysis results..."
      }
    }
  ],
  "status": "success"
}
```

**Response (Streaming):**
```
data: {"content": "Based on your symptoms"}

data: {"content": ", I'll analyze them"}

data: {"content": " and provide information..."}

data: {"content": "[DONE]"}
```

### RKLLM Generation (Optional)

**POST** `/generate`

Generate text using the RKLLM model (if available).

**Request Body:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Your message here"
    }
  ],
  "prompt": "Your prompt here (alternative to messages)",
  "role": "user",
  "enable_thinking": true,
  "stream": false,
  "add_generation_prompt": true,
  "bos_token": "REDACTED_SPECIAL_TOKEN"
}
```

## Example Usage

### Python Client Example

```python
from agent_client import AgentClient

# Initialize client
client = AgentClient("http://localhost:5000")

# Health check
health = client.health_check()
print(health)

# Send a message (system prompt is automatically included)
result = client.send_message("I have chest pain and shortness of breath. Can you help?")
print(result)

# Customize system prompt and medical tools
custom_medical_tools = [
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
        "name": "check_blood_pressure",
        "description": "Check if blood pressure is within normal range.",
        "parameters": {
            "systolic": {
                "description": "Systolic blood pressure in mmHg.",
                "type": "int",
                "default": 120
            },
            "diastolic": {
                "description": "Diastolic blood pressure in mmHg.",
                "type": "int",
                "default": 80
            }
        }
    }
]

# Set custom tools
client.set_medical_tools(custom_medical_tools)

# Send message with custom configuration
result = client.send_message("I have a headache and my blood pressure is 135/85 mmHg.")
print(result)

# Voice and TTS Example
print("\n" + "="*50)
print("Voice and TTS Example")
print("="*50)

# Get voice input
voice_text = client.get_voice_input()
if voice_text:
    print(f"Voice input: {voice_text}")
    # Send with TTS enabled
    result = client.send_message(voice_text, use_tts=True)
else:
    print("Voice input failed, using text input")
    result = client.send_message("I have a headache", use_tts=True)
```

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:5000/health

# Send message (non-streaming)
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "I have a headache and fever"
      }
    ],
    "stream": false,
    "use_tts": false
  }'

# Send message with TTS
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "I have a headache and fever"
      }
    ],
    "stream": false,
    "use_tts": true
  }'

# Send message (streaming)
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/plain" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Check my heart rate - it is 85 bpm"
      }
    ],
    "stream": true
  }'
```

## Agent Workflow

The health agent follows this workflow with optional TTS narration throughout:

1. **Planning**: Analyzes user input and creates a plan of functions to execute
   - TTS: "I'm analyzing your request and creating a plan to help you."
   - TTS: "I'll be executing X function(s) to help you."
2. **Execution**: Executes the planned functions (symptom checking, medication lookup, etc.)
   - TTS: "Now I'm executing the planned functions to gather information for you."
   - TTS: "I'm now executing the [function_name] function." (for each function)
   - TTS: "All functions have been executed successfully. Now I'll create a comprehensive summary for you."
3. **Summarization**: Creates a comprehensive medical summary of the results
   - TTS: "I'm now creating a comprehensive medical summary based on the gathered information."
   - TTS: "I've completed the analysis and created a comprehensive summary for you."
4. **Response**: Returns the final response to the user
   - TTS: "Here's my response to your request."

### Workflow Diagram

```
START → PLANNING_AGENT → [Router] → EXECUTE_PLAN → SUMMARIZE → RESPOND → END
                              ↓
                           RESPOND → END
```

## Error Handling

The server includes comprehensive error handling:

- **400 Bad Request**: Invalid JSON or missing required fields
- **503 Service Unavailable**: Server busy processing another request
- **500 Internal Server Error**: Unexpected errors during processing

## Logging

The server logs all activities to:
- Console output
- `agent_streaming_server.log` file

Log levels: INFO, WARNING, ERROR

## Performance Considerations

- **Threading**: Uses threading for non-blocking request handling
- **Resource Management**: Proper cleanup of model resources
- **Streaming**: Efficient streaming responses for real-time interaction
- **Memory**: Models are loaded once and reused across requests

## Security Notes

- The server accepts requests from any IP by default (0.0.0.0)
- Consider implementing authentication for production use
- Google Calendar credentials should be kept secure
- API keys should be stored in environment variables

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if port 5000 is available
2. **Model loading fails**: Ensure RKLLM library is available and model path is correct
3. **Google Calendar errors**: Verify credentials.json is present and valid
4. **Tavily search fails**: Check TAVILY_API_KEY environment variable
5. **Voice recognition fails**: Ensure PyAudio is installed, microphone is accessible, and Vosk model is downloaded
6. **TTS not working**: Check if pyttsx3 is installed and audio output is available
7. **Microphone access denied**: Grant microphone permissions to the application
8. **Vosk model not found**: Download and extract the Vosk model to the project directory

### Debug Mode

Run with `--debug` flag for detailed error information:

```bash
python agent_streaming_server.py --debug
```

## File Structure

```
├── agent_streaming_server.py    # Main server file
├── agent_client.py              # Client for testing
├── agent_singleton.py           # Singleton agent implementation
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables
├── credentials.json            # Google Calendar credentials
├── vosk-model-small-en-us-0.15/ # Vosk speech recognition model
├── chat_template.jinja         # Chat template for message formatting
├── agent_streaming_server.log  # Server logs
└── agent_graph.png             # Generated workflow diagram
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

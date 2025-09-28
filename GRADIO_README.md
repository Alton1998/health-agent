# Health Agent Gradio Interface

A web-based interface for testing and configuring the health agent model with custom system prompts, health management tools, and user queries.

## Features

- **Model Loading**: Load different Hugging Face models
- **System Prompt Configuration**: Customize the system prompt with placeholders
- **Health Management Tools**: Define and validate comprehensive health management tools as JSON
- **Interactive Testing**: Test queries with real-time response generation
- **Streaming Responses**: Real-time streaming using TextIterator for immediate feedback
- **Function Call Extraction**: Automatically extract and display function calls from responses
- **Example Prompts**: Quick access to common health management query examples

## Installation

1. Install the required dependencies:
```bash
pip install -r gradio_requirements.txt
```

2. Run the Gradio interface:
```bash
python gradio_health_agent.py
```

3. Open your browser and navigate to `http://localhost:7860`

## Usage

### 1. Load Model
- Enter the Hugging Face model name (default: `aldsouza/health-agent-lora-rkllm-compatible`)
- Click "Load Model" to initialize the model
- Wait for the "Model loaded successfully!" confirmation

### 2. Configure System Prompt
- Modify the system prompt in the text area
- Use `{medical_tools_json}` placeholder to insert the medical tools JSON
- The default prompt includes instructions for function calling

### 3. Define Health Management Tools
- Edit the health management tools JSON in the text area
- Click "Validate JSON" to check for syntax errors
- Each tool should have:
  - `name`: Function name
  - `description`: What the function does
  - `parameters`: Input parameters with types and defaults

### 4. Set Generation Parameters
- **Enable Streaming**: Toggle real-time streaming using TextIterator (recommended)
- **Max New Tokens**: Maximum number of tokens to generate (100-4096)
- **Temperature**: Controls randomness (0.1-2.0, higher = more random)

### 5. Test Queries
- Enter your medical query in the "User Query" field
- Use example buttons for quick testing
- Click "Generate Response" to get the model's output

## Example Health Management Tools

The interface comes with pre-configured health management tools including:

### Schedule Management
- `schedule_create_routine`: Create health routines with activities and timing
- `schedule_edit_routine`: Edit existing health routines
- `schedule_list_routines`: List all available health routines
- `schedule_update_meal_times`: Update meal times for medication scheduling
- `schedule_list_medications`: List medications and their scheduled times

### Appointment Management
- `appointment_create`: Create new appointments with healthcare providers
- `appointment_get_slots`: Get available appointment slots
- `appointment_list_upcoming`: List upcoming appointments
- `appointment_update`: Update existing appointments
- `appointment_cancel`: Cancel appointments

### Communication Tools
- `message_send`: Send messages to healthcare providers
- `message_attach`: Attach files to messages
- `careteam_get`: Get care team information

### Documentation Tools
- `camera_device_open_capture`: Capture photos for medical documentation
- `camera_five_in_one_open_capture`: Use specialized medical camera

## Example Queries

- **Health Routine**: "Create a morning health routine for patient_001 that includes blood pressure monitoring at 8:00 AM, morning medication at 8:30 AM, and light exercise at 9:00 AM."
- **Appointment Scheduling**: "Schedule a follow-up appointment with Dr. Smith (provider_001) for next Tuesday at 2:00 PM to discuss the patient's recent health improvements."
- **Care Team Communication**: "Send a message to the care team letting them know the new routine has been established and attach the patient's latest health report from last week."
- **Schedule Management**: "List all available health routines for patient_001 and update meal times to breakfast at 8:00 AM, lunch at 12:30 PM, and dinner at 6:00 PM."
- **Medical Documentation**: "Open the five-in-one medical camera and capture comprehensive medical images for patient_001 in full exam mode for documentation purposes."

## Response Format

The interface automatically:
- **Streaming Mode**: Shows real-time token generation as it happens
- **Non-Streaming Mode**: Displays complete response after generation
- Extracts and highlights function calls
- Shows generation time
- Validates JSON syntax

### Streaming Features
- **Real-time Updates**: See tokens being generated as they're produced
- **Immediate Feedback**: No waiting for complete generation
- **Progress Indication**: Visual feedback during long generations
- **TextIterator Integration**: Uses Hugging Face's TextIteratorStreamer for optimal performance

## Troubleshooting

- **Model Loading Issues**: Ensure you have sufficient GPU memory and the model name is correct
- **JSON Validation Errors**: Check the health management tools JSON syntax
- **Generation Errors**: Try reducing max_tokens or adjusting temperature
- **CUDA Errors**: Make sure PyTorch is installed with CUDA support

## Customization

You can easily customize:
- Default model name
- System prompt template
- Health management tools configuration
- Example prompts
- UI theme and layout

## API Integration

The interface can be extended to integrate with:
- Real medical databases
- Appointment scheduling systems
- Lab result APIs
- Medical device interfaces

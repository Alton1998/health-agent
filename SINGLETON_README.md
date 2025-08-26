# Health Agent Singleton Pattern

This implementation provides a singleton pattern for the compiled health agent, ensuring that only one instance of the agent (including model, tokenizer, and compiled graph) is created and accessed throughout your application.

## Overview

The singleton pattern ensures:
- **Single Instance**: Only one instance of the health agent is created
- **Lazy Initialization**: The agent is only initialized when first accessed
- **Global Access**: The same instance can be accessed from anywhere in your application
- **Resource Efficiency**: Prevents multiple model loads and memory usage

## Available Tools

The health agent provides five core medical tools:
1. **symptom_checker** - Analyze symptoms and provide possible conditions
2. **medication_lookup** - Look up details about medications
3. **book_appointment** - Schedule medical appointments
4. **check_heart_rate** - Check if heart rate is within normal range (60-100 bpm)
5. **check_temperature** - Check if body temperature is within normal range (97.0-99.5°F)

## Files

- `agent_singleton.py` - Main singleton implementation
- `example_usage.py` - Example demonstrating how to use the singleton
- `SINGLETON_README.md` - This documentation file

## Usage

### Basic Usage

```python
from agent_singleton import get_health_agent

# Get the singleton instance
agent = get_health_agent()

# Access the compiled graph
compiled_graph = agent.get_compiled_graph()

# Access model and tokenizer if needed
model = agent.get_model()
tokenizer = agent.get_tokenizer()
```

### Singleton Behavior

```python
# Multiple calls return the same instance
agent1 = get_health_agent()
agent2 = get_health_agent()

print(agent1 is agent2)  # True - same instance
```

### Using the Compiled Graph

```python
from agent_singleton import get_health_agent

# Get the agent instance
agent = get_health_agent()

# Get the compiled graph
graph = agent.get_compiled_graph()

# Create a state for execution
state = {
    "messages": [
        {"role": "user", "content": "I have a headache and fever. What could this be? Check my heart rate - it's 90 bpm. Also check my temperature - it's 98.8°F."}
    ],
    "plan": [],
    "task": "symptom_analysis"
}

# Execute the graph (this will trigger the planning and execution flow)
result = graph.invoke(state)
```

## Key Features

### 1. Singleton Class (`HealthAgentSingleton`)

- **Thread-safe**: Uses class-level variables to ensure single instance
- **Lazy initialization**: Agent components are only created when first accessed
- **Resource management**: Efficiently manages model, tokenizer, and graph instances

### 2. Builder Function (`get_health_agent()`)

- **Simple interface**: Easy-to-use function to access the singleton
- **Consistent access**: Always returns the same instance
- **Initialization handling**: Automatically initializes the agent if needed

### 3. Access Methods

- `get_compiled_graph()` - Returns the compiled LangGraph
- `get_model()` - Returns the loaded model
- `get_tokenizer()` - Returns the tokenizer
- `is_initialized()` - Checks if the agent is initialized

## Benefits

1. **Memory Efficiency**: Only one model instance loaded in memory
2. **Performance**: No repeated model loading
3. **Consistency**: Same agent instance used throughout the application
4. **Simplicity**: Easy to access from any part of your code
5. **Resource Management**: Automatic initialization and cleanup

## Example Integration

### In a Flask API

```python
from flask import Flask, request, jsonify
from agent_singleton import get_health_agent

app = Flask(__name__)

@app.route('/health-query', methods=['POST'])
def health_query():
    # Get the singleton agent instance
    agent = get_health_agent()
    
    # Get user query from request
    user_message = request.json.get('message')
    
    # Create state for graph execution
    state = {
        "messages": [{"role": "user", "content": user_message}],
        "plan": [],
        "task": "health_analysis"
    }
    
    # Execute the graph
    result = agent.get_compiled_graph().invoke(state)
    
    return jsonify(result)
```

### In a Chat Interface

```python
from agent_singleton import get_health_agent

def chat_with_agent():
    agent = get_health_agent()
    graph = agent.get_compiled_graph()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "plan": [],
            "task": "conversation"
        }
        
        result = graph.invoke(state)
        print(f"Agent: {result}")
```

## Running the Example

```bash
python example_usage.py
```

This will demonstrate:
- Singleton instance creation
- Access to compiled graph, model, and tokenizer
- Singleton behavior verification
- Basic graph setup

## Notes

- The agent requires CUDA for model execution
- Google Calendar integration requires proper credentials setup
- Functions are executed automatically without user approval
- Make sure all required dependencies are installed

## Dependencies

Ensure you have all the required packages from your original code:
- torch
- transformers
- langgraph
- langchain_community
- pytz
- tzlocal
- google-auth-oauthlib
- google-api-python-client
- python-dotenv
- regex

The singleton pattern provides a clean, efficient way to manage your health agent throughout your application while ensuring optimal resource usage.

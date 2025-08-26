import requests
import json
import time

# Configuration
SERVER_URL = "http://10.0.0.17:8080"

# Medical tools from voice-ai-agent.py
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
    }, 
    {
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
    }, 
    {
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

# System prompt from voice-ai-agent.py
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

def test_connection():
    """Test connection to server"""
    print("Testing connection to server...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server connection successful!")
            health_data = response.json()
            print(f"  Server status: {health_data.get('status', 'unknown')}")
            print(f"  Model loaded: {health_data.get('model_loaded', 'unknown')}")
            return True
        else:
            print(f"✗ Server returned status: {response.status_code}")
            return False
    except requests.exceptions.ConnectTimeout:
        print("✗ Connection timeout - server not reachable")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Connection error - check if server is running")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def make_request_with_retry(url, data, max_retries=3, delay=5):
    """Make a request with retry logic for busy server"""
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            response = requests.post(url, json=data, timeout=200)
            
            if response.status_code == 503:
                error_data = response.json()
                if "busy" in error_data.get("message", "").lower():
                    if attempt < max_retries - 1:
                        print(f"    Server busy, waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"    Server still busy after {max_retries} attempts")
                        return response
            
            return response
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"    Request timed out, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Request failed: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                raise
    
    return None

def test_blood_pressure_prompt():
    """Test the blood pressure prompt format with medical tools"""
    print("=" * 60)
    print("Testing Blood Pressure Prompt Format with Medical Tools")
    print("=" * 60)
    
    # Define the blood pressure prompt in the exact format you specified
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
    
    print("System Prompt (truncated):")
    print(f"  {system_prompt[:100]}...")
    print()
    print("User Message:")
    print(f"  {blood_pressure_prompt[1]['content']}")
    print()
    
    # Test non-streaming chat
    print("1. Testing Non-Streaming Chat...")
    data = {
        "messages": blood_pressure_prompt,
        "stream": False,
        "add_generation_prompt": True,
    }
    
    try:
        response = make_request_with_retry(f"{SERVER_URL}/chat", data)
        if response is None:
            print("✗ All retry attempts failed")
            return
            
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✓ Chat successful!")
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print("Assistant response:")
                print(f"  {content}")
            else:
                print("✗ No response content found")
        else:
            print(f"✗ Chat failed: {response.text}")
    except Exception as e:
        print(f"✗ Chat error: {e}")
    
    print()
    
    # Add delay between tests to avoid overwhelming the server
    print("Waiting 10 seconds before streaming test...")
    time.sleep(10)
    
    # Test streaming chat
    print("2. Testing Streaming Chat...")
    data = {
        "messages": blood_pressure_prompt,
        "stream": True,
        "add_generation_prompt": True,
        "stream_granularity": "token"  # Stream each token as generated by model
    }
    
    try:
        response = make_request_with_retry(f"{SERVER_URL}/chat", data)
        if response is None:
            print("✗ All retry attempts failed")
            return
            
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✓ Streaming chat started:")
            print("-" * 40)
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        try:
                            data_json = json.loads(data_str)
                            content = data_json.get('content', '')
                            if content == '[DONE]':
                                print("\n--- Streaming completed ---")
                                break
                            else:
                                print(content, end='', flush=True)
                                full_response += content
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data_str}")
            
            print(f"\n\nFull streaming response:")
            print(f"  {full_response}")
        else:
            print(f"✗ Streaming chat failed: {response.text}")
    except Exception as e:
        print(f"✗ Streaming chat error: {e}")
    
    print()

def main():
    """Run the blood pressure prompt test"""
    print("Blood Pressure Prompt Test Client")
    print("=" * 80)
    print(f"Testing with {len(medical_tools)} medical tools")
    print("=" * 80)
    
    # Test connection first
    if not test_connection():
        print("Cannot connect to server. Please check:")
        print("1. Server is running")
        print("2. Server is accessible on the network")
        print("3. Firewall allows connections to port 5000")
        print("4. Try using 'localhost' instead of IP address")
        return
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(5)
    
    # Run test
    test_blood_pressure_prompt()
    
    print("=" * 80)
    print("Blood pressure prompt test completed!")

if __name__ == "__main__":
    main()

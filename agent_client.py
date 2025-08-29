import requests
import json
import time
import sys
import speech_recognition as sr
import threading
import queue
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentClient:
    def __init__(self, base_url="http://10.37.101.57:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        logger.info(f"AgentClient initialized with base URL: {base_url}")
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.listen_thread = None
        logger.info("Speech recognition components initialized")
        
        # Vosk will automatically look for a "model" directory
        logger.info("Vosk will automatically use the 'model' directory if available")
        
        # Adjust for ambient noise
        logger.info("Adjusting for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        logger.info("Ambient noise adjustment completed")
        
        # Medical tools configuration
        self.medical_tools = [
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
                "description": "Check if body temperature is within normal range (97.0-99.5°F).",
                "parameters": {
                    "temperature": {
                        "description": "Body temperature in Fahrenheit (°F).",
                        "type": "float",
                        "default": 98.6
                    }
                }
            }
        ]
        
        # System prompt
        self.system_prompt = f"""
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
{json.dumps(self.medical_tools, indent=2)}
"""
    
    def health_check(self):
        """Check if the server is healthy"""
        try:
            logger.info(f"Checking server health at {self.base_url}/health")
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            health_data = response.json()
            logger.info(f"Server health check successful: {health_data}")
            return health_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            print(f"Health check failed: {e}")
            return None
    
    def set_system_prompt(self, system_prompt):
        """Set a custom system prompt"""
        logger.info(f"Setting custom system prompt (length: {len(system_prompt)})")
        logger.debug(f"System prompt: {repr(system_prompt)}")
        self.system_prompt = system_prompt
    
    def set_medical_tools(self, medical_tools):
        """Set custom medical tools configuration"""
        logger.info(f"Setting custom medical tools configuration with {len(medical_tools)} tools")
        logger.debug(f"Medical tools: {json.dumps(medical_tools, indent=2)}")
        self.medical_tools = medical_tools
        # Update the system prompt to include the new tools
        self.system_prompt = f"""
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
{json.dumps(self.medical_tools, indent=2)}
"""
        logger.info("System prompt updated with new medical tools")
    
    def get_medical_tools(self):
        """Get the current medical tools configuration"""
        return self.medical_tools
    
    def get_system_prompt(self):
        """Get the current system prompt"""
        return self.system_prompt
    
    def set_vosk_model_path(self, model_path):
        """Set a custom Vosk model path (if needed)"""
        logger.info(f"Vosk model path set to: {model_path}")
        self.recognizer.vosk_model_path = model_path
    
    def get_vosk_model_path(self):
        """Get the current Vosk model path"""
        return getattr(self.recognizer, 'vosk_model_path', 'model')
    
    def start_voice_listening(self):
        """Start listening for voice input in background"""
        if self.is_listening:
            logger.warning("Already listening for voice input")
            print("Already listening for voice input")
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_for_audio, daemon=True)
        self.listen_thread.start()
        logger.info("Background voice listening started")
        print("Voice listening started. Speak to provide input.")
    
    def stop_voice_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1)
        logger.info("Background voice listening stopped")
        print("Voice listening stopped.")
    
    def _listen_for_audio(self):
        """Background thread for listening to audio"""
        logger.info("Background audio listening thread started")
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio)
                    logger.debug("Audio captured and queued")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error listening for audio: {e}")
                print(f"Error listening for audio: {e}")
                break
        logger.info("Background audio listening thread stopped")
    
    def get_voice_input(self, delay_seconds=5):
        """Get voice input and convert to text using Vosk"""
        try:
            logger.info(f"Starting voice input capture with {delay_seconds} second delay")
            
            if delay_seconds > 0:
                print(f"Voice recording will start in {delay_seconds} seconds...")
                print("Please prepare to speak.")
                
                # Countdown
                for i in range(delay_seconds, 0, -1):
                    print(f"Recording starts in {i}...")
                    time.sleep(1)
            else:
                print("Starting voice recording immediately...")
            
            print("Listening... Speak now.")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            logger.info("Processing speech with Vosk...")
            print("Processing speech with Vosk...")
            text = self.recognizer.recognize_vosk(audio)
            logger.info(f"Voice recognition successful: {repr(text)}")
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            print("Could not understand audio")
            return None
        except Exception as e:
            logger.error(f"Error in voice recognition: {e}")
            print(f"Error in voice recognition: {e}")
            return None
    
    def get_queued_voice_input(self):
        """Get voice input from the background queue using Vosk"""
        try:
            audio = self.audio_queue.get(timeout=1)
            logger.info("Processing queued audio with Vosk...")
            text = self.recognizer.recognize_vosk(audio)
            logger.info(f"Queued voice recognition successful: {repr(text)}")
            print(f"Recognized: {text}")
            return text
        except queue.Empty:
            logger.debug("No audio in queue")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand queued audio")
            print("Could not understand audio")
            return None
        except Exception as e:
            logger.error(f"Error in queued voice recognition: {e}")
            print(f"Error in voice recognition: {e}")
            return None
    
    def send_message(self, message, stream=False, use_tts=False):
        """Send a message to the agent and get response"""
        try:
            logger.info(f"Sending message to agent (stream: {stream}, TTS: {use_tts})")
            logger.info(f"User message: {repr(message)}")
            
            # Prepare the message with system prompt
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
            
            logger.debug(f"Prepared messages: {json.dumps(messages, indent=2)}")
            
            payload = {
                "messages": messages,
                "stream": stream,
                "use_tts": use_tts
            }
            
            if stream:
                # Streaming response
                logger.info("Sending streaming request to server")
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    stream=True,
                    headers={'Accept': 'text/plain'}
                )
                response.raise_for_status()
                
                print("Streaming response:")
                logger.info("Starting to receive streaming response")
                chunk_count = 0
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            try:
                                data = json.loads(data_str)
                                if data.get('content') == '[DONE]':
                                    logger.info("Streaming response completed")
                                    break
                                elif 'error' in data:
                                    logger.error(f"Streaming error: {data['error']}")
                                    print(f"Error: {data['error']}")
                                    break
                                else:
                                    content = data.get('content', '')
                                    chunk_count += 1
                                    logger.debug(f"Streaming chunk {chunk_count}: {repr(content)}")
                                    if chunk_count % 10 == 0:  # Log every 10 chunks
                                        logger.info(f"Streaming chunk {chunk_count}, content: {repr(content[:50])}...")
                                    print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                logger.error(f"Invalid JSON in streaming response: {data_str}")
                                print(f"Invalid JSON: {data_str}")
                
                logger.info(f"Streaming completed, total chunks: {chunk_count}")
                print("\n--- End of streaming response ---")
                
            else:
                # Non-streaming response
                response = self.session.post(f"{self.base_url}/chat", json=payload)
                response.raise_for_status()
                
                result = response.json()
                print("Response:")
                print(json.dumps(result, indent=2))
                
                # Print the final message content
                messages = result.get("messages", [])
                if messages:
                    final_message = messages[-1]
                    if final_message.get("role") == "assistant":
                        print(f"\nFinal Response: {final_message.get('content', '')}")
                
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def send_messages(self, messages_list, stream=False, use_tts=False):
        """Send a list of messages to the agent"""
        try:
            logger.info(f"Sending messages list to agent (stream: {stream}, TTS: {use_tts})")
            logger.info(f"Messages count: {len(messages_list)}")
            logger.debug(f"Messages list: {json.dumps(messages_list, indent=2)}")
            
            payload = {
                "messages": messages_list,
                "stream": stream,
                "use_tts": use_tts
            }
            
            if stream:
                # Streaming response
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    stream=True,
                    headers={'Accept': 'text/plain'}
                )
                response.raise_for_status()
                
                print("Streaming response:")
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            try:
                                data = json.loads(data_str)
                                if data.get('content') == '[DONE]':
                                    break
                                elif 'error' in data:
                                    print(f"Error: {data['error']}")
                                    break
                                else:
                                    print(data.get('content', ''), end='', flush=True)
                            except json.JSONDecodeError:
                                print(f"Invalid JSON: {data_str}")
                
                print("\n--- End of streaming response ---")
                
            else:
                # Non-streaming response
                response = self.session.post(f"{self.base_url}/chat", json=payload)
                response.raise_for_status()
                
                result = response.json()
                print("Response:")
                print(json.dumps(result, indent=2))
                
                # Print the final message content
                messages = result.get("messages", [])
                if messages:
                    final_message = messages[-1]
                    if final_message.get("role") == "assistant":
                        print(f"\nFinal Response: {final_message.get('content', '')}")
                
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

def main():
    logger.info("Starting Agent Client - Voice Only Mode")
    
    # Initialize client
    client = AgentClient()
    
    # Check server health
    print("Checking server health...")
    logger.info("Performing initial server health check")
    health = client.health_check()
    if health:
        logger.info("Server health check successful")
        print(f"Server is healthy: {json.dumps(health, indent=2)}")
    else:
        logger.error("Server health check failed")
        print("Server is not responding. Make sure the server is running.")
        return
    
    print("\n" + "="*60)
    print("Health Agent Client - Voice Only Mode")
    print("="*60)
    print("This is a voice-controlled health agent.")
    print("Press 'r' to start recording your voice query.")
    print("Press 'q' to quit the application.")
    print("="*60)
    
    # Voice-only interaction loop
    voice_interaction_loop(client)
    
    print("\n" + "="*60)
    print("Voice Agent Session Ended")
    print("="*60)

def voice_interaction_loop(client):
    """Voice-only interaction loop with 'r' to record and 'q' to quit"""
    logger.info("Starting voice interaction loop")
    
    print("\nVoice Agent Ready!")
    print("Available commands:")
    print("- Press 'r' to start recording your voice query")
    print("- Press 'q' to quit the application")
    print("- Press 'h' to see available medical tools")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter command (r/q/h): ").strip().lower()
            
            if user_input == 'q':
                logger.info("User requested to quit")
                print("Goodbye!")
                break
                
            elif user_input == 'h':
                logger.info("User requested help")
                print("\n" + "="*50)
                print("AVAILABLE MEDICAL TOOLS")
                print("="*50)
                tools = client.get_medical_tools()
                for i, tool in enumerate(tools, 1):
                    print(f"{i}. {tool['name'].replace('_', ' ').title()}")
                    print(f"   Description: {tool['description']}")
                    print(f"   Parameters:")
                    for param_name, param_info in tool['parameters'].items():
                        print(f"     - {param_name}: {param_info['description']} (Type: {param_info['type']}, Default: {param_info['default']})")
                    print()
                print("Example voice queries:")
                print("- 'I have a headache and fever, what could be wrong?'")
                print("- 'Tell me about Ibuprofen medication'")
                print("- 'My heart rate is 85 bpm, is that normal?'")
                print("- 'My temperature is 99.2°F, should I be concerned?'")
                print("="*50)
                
            elif user_input == 'r':
                logger.info("User requested voice recording")
                print("\n" + "="*50)
                print("VOICE RECORDING")
                print("="*50)
                print("Starting voice recording...")
                print("Please speak your medical query clearly.")
                print("Recording will automatically stop after 10 seconds or when you stop speaking.")
                
                # Get voice input without delay
                voice_text = client.get_voice_input(delay_seconds=0)
                
                if voice_text:
                    logger.info(f"Voice input received: {repr(voice_text)}")
                    print(f"\nRecognized: '{voice_text}'")
                    print("\nProcessing your query...")
                    
                    # Send to agent with TTS enabled
                    logger.info("Sending voice input to agent with TTS enabled")
                    result = client.send_message(voice_text, use_tts=True)
                    
                    if result:
                        print("\n" + "="*50)
                        print("RESPONSE COMPLETED")
                        print("="*50)
                    else:
                        print("\nError: Failed to get response from agent")
                else:
                    logger.warning("Voice input failed")
                    print("\nCould not understand your voice input.")
                    print("Please try again and speak more clearly.")
                
            else:
                print("Invalid command. Use 'r' to record, 'h' for help, or 'q' to quit.")
                
        except KeyboardInterrupt:
            logger.info("User interrupted with Ctrl+C")
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in voice interaction loop: {e}")
            print(f"\nError: {e}")
            print("Please try again.")

def run_example():
    """Run example requests demonstrating available tools"""
    logger.info("Starting example requests")
    client = AgentClient()
    
    # Check health
    print("Checking server health...")
    logger.info("Performing server health check for examples")
    health = client.health_check()
    if not health:
        logger.error("Server health check failed for examples")
        print("Server is not responding. Make sure the server is running.")
        return
    
    logger.info("Server health check successful for examples")
    print(f"Server is healthy: {json.dumps(health, indent=2)}")
    
    # Run voice interaction loop
    voice_interaction_loop(client)
    
    print("\n" + "="*60)
    print("VOICE EXAMPLE COMPLETED")
    print("="*60)
    print("The voice-controlled health agent demonstrated:")
    print("✓ Voice Input - Speech recognition using Vosk")
    print("✓ Text-to-Speech - Audio output for responses")
    print("✓ Medical Tools - Symptom checker, medication lookup, vitals check")
    print("✓ Interactive Commands - 'r' to record, 'h' for help, 'q' to quit")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        run_example()
    else:
        main()

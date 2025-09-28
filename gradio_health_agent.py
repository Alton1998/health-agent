import json
import time
import torch
import gradio as gr
from regex import regex
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model
model = None
tokenizer = None
model_loaded = False

# Health management tools from the original file
default_health_management_tools = [
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

# Default system prompt
default_system_prompt = """You are an intelligent AI assistant that uses available tools (functions) to help users achieve their medical-related goals. Your job is to understand the user's intent, identify missing information if needed, and then select and call the most appropriate function(s) to solve the task.

# Rules:
- ALWAYS use the tools provided to answer the user's request, unless explicitly told not to.
- Ask clarifying questions ONLY if the user's request is ambiguous or lacks required input parameters.
- If multiple tools are needed, use them in sequence.
- DO NOT make up data or assume values — request any missing input clearly.

# Output Format:
- Respond using a JSON list of function calls in the following format:
  [
    {
      "name": "function_name",
      "arguments": {
        "param1": "value1",
        "param2": "value2"
      }
    }
  ]
- Only include the functions needed to complete the task.
- If no function is needed or the input is unclear, ask a clarifying question instead of guessing.
- Do NOT respond with explanations or natural language outside the JSON block unless explicitly instructed.

Following are the tools provided to you:
{medical_tools_json}"""

def load_model(model_name="aldsouza/health-agent-lora-rkllm-compatible"):
    """Load the model and tokenizer"""
    global model, tokenizer, model_loaded
    
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        model_loaded = True
        print("Model loaded successfully!")
        return "✅ Model loaded successfully!"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"❌ Error loading model: {str(e)}"

def generate_response_streaming(system_prompt, medical_tools_json, user_prompt, max_tokens, temperature):
    """Generate streaming response using the loaded model with real-time streaming"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        yield "❌ Please load the model first!"
        return
    
    try:
        # Parse medical tools JSON
        try:
            medical_tools = json.loads(medical_tools_json)
        except json.JSONDecodeError as e:
            yield f"❌ Invalid JSON in medical tools: {str(e)}"
            return
        
        # Create system prompt with tools
        # Escape the JSON properly for the format string
        medical_tools_str = json.dumps(medical_tools, indent=2)
        full_system_prompt = system_prompt.replace("{medical_tools_json}", medical_tools_str)
        
        # Create messages
        messages = [
            {
                "content": full_system_prompt,
                "role": "system"
            },
            {
                "content": user_prompt,
                "role": "user"
            }
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response with streaming
        start_time = time.time()
        
        # Use TextIterator for streaming
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=30.0, 
            skip_special_tokens=True, 
            skip_prompt=True
        )
        
        # Generation parameters
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the response in real-time
        assistant_response = ""
        yield "🚀 Starting generation...\n\n"
        
        for new_text in streamer:
            assistant_response += new_text
            yield assistant_response
        
        # Wait for thread to complete
        thread.join()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Try to extract function calls using regex
        pattern = r'''
        \{                      # Opening brace of the function block
        \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
        \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
        (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
        \})                                # Closing brace of arguments
        \s*\}                             # Closing brace of the function block
        '''
        
        matches = regex.findall(pattern, assistant_response, regex.VERBOSE)
        
        # Add final analysis
        final_result = assistant_response + f"\n\n=== GENERATION COMPLETE ===\n"
        final_result += f"Generation Time: {generation_time:.2f} seconds\n\n"
        
        if matches:
            final_result += f"Extracted Function Calls ({len(matches)}):\n"
            for i, (func_name, args_json) in enumerate(matches, 1):
                final_result += f"\nFunction {i}: {func_name}\n"
                final_result += f"Arguments: {args_json}\n"
        else:
            final_result += "No function calls detected in response.\n"
        
        yield final_result
        
    except Exception as e:
        yield f"❌ Error generating response: {str(e)}"

def generate_response(system_prompt, medical_tools_json, user_prompt, max_tokens, temperature):
    """Generate response using the loaded model (non-streaming version)"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        return "❌ Please load the model first!"
    
    try:
        # Parse medical tools JSON
        try:
            medical_tools = json.loads(medical_tools_json)
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON in medical tools: {str(e)}"
        
        # Create system prompt with tools
        # Escape the JSON properly for the format string
        medical_tools_str = json.dumps(medical_tools, indent=2)
        full_system_prompt = system_prompt.replace("{medical_tools_json}", medical_tools_str)
        
        # Create messages
        messages = [
            {
                "content": full_system_prompt,
                "role": "system"
            },
            {
                "content": user_prompt,
                "role": "user"
            }
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove the input prompt)
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        assistant_response = response[len(input_text):].strip()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Try to extract function calls using regex
        pattern = r'''
        \{                      # Opening brace of the function block
        \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
        \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
        (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
        \})                                # Closing brace of arguments
        \s*\}                             # Closing brace of the function block
        '''
        
        matches = regex.findall(pattern, assistant_response, regex.VERBOSE)
        
        # Format the response
        result = f"Generation Time: {generation_time:.2f} seconds\n\n"
        result += f"Full Response:\n{assistant_response}\n\n"
        
        if matches:
            result += f"Extracted Function Calls ({len(matches)}):\n"
            for i, (func_name, args_json) in enumerate(matches, 1):
                result += f"\nFunction {i}: {func_name}\n"
                result += f"Arguments: {args_json}\n"
        else:
            result += "No function calls detected in response.\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Health Agent Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Health Agent Interface")
        gr.Markdown("Configure and test your health agent with custom system prompts, medical tools, and user queries.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔧 Configuration")
                
                # Model loading section
                with gr.Group():
                    gr.Markdown("### Model Loading")
                    model_name_input = gr.Textbox(
                        label="Model Name",
                        value="aldsouza/health-agent-lora-rkllm-compatible",
                        placeholder="Enter Hugging Face model name"
                    )
                    load_model_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Model Status", interactive=False)
                
                # System prompt section
                with gr.Group():
                    gr.Markdown("### System Prompt")
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value=default_system_prompt,
                        lines=10,
                        placeholder="Enter your system prompt. Use {medical_tools_json} to insert tools."
                    )
                
                # Health management tools section
                with gr.Group():
                    gr.Markdown("### Health Management Tools (JSON)")
                    medical_tools_input = gr.Textbox(
                        label="Health Management Tools JSON",
                        value=json.dumps(default_health_management_tools, indent=2),
                        lines=15,
                        placeholder="Enter health management tools as JSON array"
                    )
                    validate_json_btn = gr.Button("Validate JSON", variant="secondary")
                    json_validation = gr.Textbox(label="JSON Validation", interactive=False)
                
                # Generation parameters
                with gr.Group():
                    gr.Markdown("### Generation Parameters")
                    streaming_enabled = gr.Checkbox(
                        label="Enable Streaming",
                        value=True,
                        info="Stream responses in real-time using TextIterator"
                    )
                    max_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=100,
                        maximum=4096,
                        value=1024,
                        step=100
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("## 💬 User Query & Response")
                
                # User prompt section
                with gr.Group():
                    gr.Markdown("### User Query")
                    user_prompt_input = gr.Textbox(
                        label="User Prompt",
                        value="Create a morning health routine for patient_001 that includes blood pressure monitoring at 8:00 AM, morning medication at 8:30 AM, and light exercise at 9:00 AM. Then schedule a follow-up appointment with Dr. Smith (provider_001) for next Tuesday at 2:00 PM to discuss the patient's recent health improvements.",
                        lines=5,
                        placeholder="Enter your health management query here..."
                    )
                    
                    # Example prompts
                    gr.Markdown("### Example Prompts")
                    example_buttons = gr.Row()
                    with example_buttons:
                        example1_btn = gr.Button("Health Routine", size="sm")
                        example2_btn = gr.Button("Appointment", size="sm")
                        example3_btn = gr.Button("Message Care Team", size="sm")
                        example4_btn = gr.Button("Schedule Management", size="sm")
                        example5_btn = gr.Button("Camera Capture", size="sm")
                
                # Generate buttons
                button_row = gr.Row()
                with button_row:
                    generate_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
                    stream_btn = gr.Button("📡 Stream Response", variant="secondary", size="lg")
                
                # Response section
                with gr.Group():
                    gr.Markdown("### Response")
                    response_output = gr.Textbox(
                        label="Generated Response", 
                        lines=20, 
                        max_lines=50,
                        interactive=False,
                        show_copy_button=True
                    )
        
        # Event handlers
        load_model_btn.click(
            fn=load_model,
            inputs=[model_name_input],
            outputs=[model_status]
        )
        
        validate_json_btn.click(
            fn=lambda x: "✅ Valid JSON" if json.loads(x) else "❌ Invalid JSON",
            inputs=[medical_tools_input],
            outputs=[json_validation]
        )
        
        def generate_with_mode(system_prompt, medical_tools_json, user_prompt, max_tokens, temperature, streaming):
            """Choose between streaming and non-streaming based on checkbox"""
            if streaming:
                # For streaming, we need to collect the generator output
                generator = generate_response_streaming(system_prompt, medical_tools_json, user_prompt, max_tokens, temperature)
                # Convert generator to string for display
                result = ""
                for chunk in generator:
                    result = chunk  # Keep the latest chunk
                return result
            else:
                # For non-streaming, return the regular response
                return generate_response(system_prompt, medical_tools_json, user_prompt, max_tokens, temperature)
        
        # Create a separate streaming function for real-time updates
        def stream_response(system_prompt, medical_tools_json, user_prompt, max_tokens, temperature):
            """Stream response with real-time updates"""
            if not model_loaded:
                yield "❌ Please load the model first!"
                return
            
            try:
                # Parse medical tools JSON
                try:
                    medical_tools = json.loads(medical_tools_json)
                except json.JSONDecodeError as e:
                    yield f"❌ Invalid JSON in medical tools: {str(e)}"
                    return
                
                # Create system prompt with tools
                medical_tools_str = json.dumps(medical_tools, indent=2)
                full_system_prompt = system_prompt.replace("{medical_tools_json}", medical_tools_str)
                
                # Create messages
                messages = [
                    {"content": full_system_prompt, "role": "system"},
                    {"content": user_prompt, "role": "user"}
                ]
                
                # Apply chat template
                inputs = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, 
                    return_dict=True, return_tensors="pt"
                ).to(model.device)
                
                # Generate response with streaming
                start_time = time.time()
                
                from transformers import TextIteratorStreamer
                from threading import Thread
                
                # Create streamer
                streamer = TextIteratorStreamer(
                    tokenizer, timeout=30.0, skip_special_tokens=True, skip_prompt=True
                )
                
                # Generation parameters
                generation_kwargs = {
                    **inputs, "max_new_tokens": max_tokens, "temperature": temperature,
                    "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "streamer": streamer
                }
                
                # Start generation in a separate thread
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream the response in real-time
                assistant_response = ""
                yield "🚀 Starting generation...\n\n"
                
                for new_text in streamer:
                    assistant_response += new_text
                    yield assistant_response
                
                # Wait for thread to complete
                thread.join()
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Try to extract function calls using regex
                pattern = r'''
                \{                      # Opening brace of the function block
                \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
                \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
                (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
                \})                                # Closing brace of arguments
                \s*\}                             # Closing brace of the function block
                '''
                
                matches = regex.findall(pattern, assistant_response, regex.VERBOSE)
                
                # Add final analysis
                final_result = assistant_response + f"\n\n=== GENERATION COMPLETE ===\n"
                final_result += f"Generation Time: {generation_time:.2f} seconds\n\n"
                
                if matches:
                    final_result += f"Extracted Function Calls ({len(matches)}):\n"
                    for i, (func_name, args_json) in enumerate(matches, 1):
                        final_result += f"\nFunction {i}: {func_name}\n"
                        final_result += f"Arguments: {args_json}\n"
                else:
                    final_result += "No function calls detected in response.\n"
                
                yield final_result
                
            except Exception as e:
                yield f"❌ Error generating response: {str(e)}"
        
        # Handle streaming vs non-streaming differently
        generate_btn.click(
            fn=generate_with_mode,
            inputs=[system_prompt_input, medical_tools_input, user_prompt_input, max_tokens, temperature, streaming_enabled],
            outputs=[response_output],
            show_progress=True
        )
        
        # Add streaming button handler
        stream_btn.click(
            fn=stream_response,
            inputs=[system_prompt_input, medical_tools_input, user_prompt_input, max_tokens, temperature],
            outputs=[response_output],
            show_progress=True
        )
        
        # Example button handlers
        example1_btn.click(
            fn=lambda: "Create a morning health routine for patient_001 that includes blood pressure monitoring at 8:00 AM, morning medication at 8:30 AM, and light exercise at 9:00 AM.",
            outputs=[user_prompt_input]
        )
        
        example2_btn.click(
            fn=lambda: "Schedule a follow-up appointment with Dr. Smith (provider_001) for next Tuesday at 2:00 PM to discuss the patient's recent health improvements.",
            outputs=[user_prompt_input]
        )
        
        example3_btn.click(
            fn=lambda: "Send a message to the care team letting them know the new routine has been established and attach the patient's latest health report from last week.",
            outputs=[user_prompt_input]
        )
        
        example4_btn.click(
            fn=lambda: "List all available health routines for patient_001 and update meal times to breakfast at 8:00 AM, lunch at 12:30 PM, and dinner at 6:00 PM.",
            outputs=[user_prompt_input]
        )
        
        example5_btn.click(
            fn=lambda: "Open the five-in-one medical camera and capture comprehensive medical images for patient_001 in full exam mode for documentation purposes.",
            outputs=[user_prompt_input]
        )
        
        # Auto-load model on startup
        demo.load(
            fn=load_model,
            inputs=[model_name_input],
            outputs=[model_status]
        )
    
    return demo

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(11)
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

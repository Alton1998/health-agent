import os
import pickle
import json
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

load_dotenv()


class HealthAgentSingleton:
    """
    Singleton class to manage the compiled health agent.
    Ensures only one instance of the agent is created and accessed.
    """
    _instance = None
    _compiled_graph = None
    _model = None
    _tokenizer = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HealthAgentSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._setup_agent()

    def _setup_agent(self):
        """Initialize the agent components (model, tokenizer, graph)"""
        print("Initializing Health Agent...")

        # Set random seed
        torch.manual_seed(11)

        # Model configuration
        model_name = "aldsouza/health-agent"

        # Initialize tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to("cuda")

        # Build and compile the graph
        self._compiled_graph = self._build_graph()

        print("Health Agent initialized successfully!")

    def _build_graph(self):
        """Build and compile the LangGraph for the health agent"""

        # Pattern for function extraction
        pattern = r'''
            \{                      # Opening brace of the function block
            \s*"name"\s*:\s*"([^"]+)"\s*,      # Capture the function name
            \s*"arguments"\s*:\s*(\{            # Capture the arguments JSON object starting brace
            (?:[^{}]++ | (?2))*?                # Recursive matching for balanced braces (PCRE syntax)
            \})                                # Closing brace of arguments
            \s*\}                             # Closing brace of the function block
            '''

        # Function execution map
        function_execution_map = {
            "symptom_checker": self._symptom_checker,
            "medication_lookup": self._medication_lookup,
            "book_appointment": self._book_appointment,
            "check_heart_rate": self._check_heart_rate,
            "check_temperature": self._check_temperature
        }

        # State definition
        class State(TypedDict):
            messages: List[Dict[str, Any]]
            plan: List[Dict[str, Any]]
            task: str
            text_obj: Any

        # Build the graph
        graph_builder = StateGraph(State)

        # Node definitions
        PLANNING_AGENT = "PLANNING_AGENT"
        EXECUTE_PLAN = "EXECUTE_PLAN"
        RESPOND = "RESPOND"
        SUMMARIZE = "SUMMARIZE"

        def planning(state: State):
            print("Coming up with Plan")
            messages = state.get("messages", [])
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True)
            generation_kwargs = dict(inputs, streamer=streamer,
                                     max_new_tokens=4096,
                                     temperature=0.7, )
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                print(new_text, end="")
                generated_text = generated_text + new_text
            generated_text = generated_text.replace("REDACTED_SPECIAL_TOKEN", "").replace("</think>", "")

            matches = regex.findall(pattern, generated_text, regex.VERBOSE)
            plan = state.get("plan", [])

            for i, (func_name, args_json) in enumerate(matches, 1):
                plan_entry = dict()
                plan_entry["function_name"] = func_name
                plan_entry["arguments"] = json.loads(args_json)
                plan.append(plan_entry)

            messages.append({"role": "assistant", "content": generated_text})

            return {"messages": messages, "plan": plan}

        def router(state: State):
            plan = state.get("plan", [])
            if len(plan) > 0:
                return "execute_plan"
            return "respond"

        def execute_plan(state: State):
            print("Executing")
            plan = state.get("plan", [])
            for plan_entry in plan:
                plan_entry["status"] = dict()
                print(f"Executing {plan_entry['function_name']} with details {plan_entry['arguments']}")
                
                if plan_entry["function_name"] in function_execution_map.keys():
                    function = function_execution_map[plan_entry["function_name"]]
                    result = function(plan_entry["arguments"])
                    plan_entry["status"] = result
                    print(f"Task completed successfully.")
                else:
                    print(f"Capability not implemented for {plan_entry['function_name']}")
                    plan_entry["status"] = {"status": 400, "message": "Function not implemented"}
                
                print("Proceeding with next.")

            return {"plan": plan}

        def respond(state: State):
            print(state.get("messages")[-1]["content"])
            return {"plan": state.get("plan")}

        def summarize(state: State):
            plan = state.get("plan")
            messages = state.get("messages")
            
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
                    # For successful executions, provide a more detailed summary
                    if function_name == "symptom_checker":
                        execution_summary.append(f"{i}. Symptom Analysis ({args_summary}) - SUCCESS: Analyzed symptoms and provided possible conditions")
                    elif function_name == "medication_lookup":
                        execution_summary.append(f"{i}. Medication Lookup ({args_summary}) - SUCCESS: Retrieved detailed information about the medication")
                    elif function_name == "book_appointment":
                        # Extract event link if available
                        event_link = "No link available"
                        if isinstance(message, dict):
                            if "event_link" in message:
                                event_link = message["event_link"]
                            elif "htmlLink" in str(message):
                                # Try to extract the link from the message
                                import re
                                # Updated regex to match both www.google.com and calendar.google.com patterns
                                link_match = re.search(r'https://(?:www\.)?google\.com/calendar/event\?[^\s]+', str(message))
                                if link_match:
                                    event_link = link_match.group(0)
                        execution_summary.append(f"{i}. Appointment Booking ({args_summary}) - SUCCESS: Scheduled medical appointment. Event Link: {event_link}")
                    elif function_name == "check_heart_rate":
                        execution_summary.append(f"{i}. Heart Rate Check ({args_summary}) - SUCCESS: Heart rate assessment completed")
                    elif function_name == "check_temperature":
                        execution_summary.append(f"{i}. Temperature Check ({args_summary}) - SUCCESS: Temperature assessment completed")
                    else:
                        execution_summary.append(f"{i}. {function_name}({args_summary}) - SUCCESS: Task completed successfully")
                else:
                    execution_summary.append(f"{i}. {function_name}({args_summary}) - FAILED (Status {status_code}): {message}")
            
            summary_prompt = []
            summary_prompt.append({
                "role": "user",
                 "content": f"""Based on the following execution results, provide a comprehensive medical summary:

{chr(10).join(execution_summary)}

IMPORTANT: Use ONLY the information provided in the execution results above. DO NOT hallucinate, make assumptions, or add information that is not explicitly stated in the results. If information is missing or unclear, state that clearly rather than making up details.

Please provide a structured summary that includes:

1. **Tasks Completed**: List what medical functions were successfully executed (based on the results above)
2. **Key Findings**: Highlight important medical information discovered from the actual results (symptoms, vitals, medication details, appointment links)
3. **Health Assessment**: Provide insights about the patient's condition based ONLY on the provided results
4. **Recommendations**: Suggest next steps or actions based on the actual findings (do not make up recommendations)
5. **Appointments**: Note any scheduled appointments, including the event link if provided

Format your response as a clear, professional medical summary that a healthcare provider would understand. Focus on actionable insights and patient care recommendations based on the actual data provided."""
             })
            inputs = self._tokenizer.apply_chat_template(
                summary_prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True)
            generation_kwargs = dict(inputs, streamer=streamer,
                                     max_new_tokens=4096,
                                     temperature=0.7, )
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                print(new_text, end="")
                generated_text = generated_text + new_text
            messages.append({"role": "assistant", "content": generated_text})

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
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Graph saved as graph.png")

        return compiled_graph

    def _symptom_checker(self, kwargs):
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

    def _medication_lookup(self, kwargs):
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

    def _create_google_calendar_meeting(self, summary: str, start_datetime: str, end_datetime: str,
                                        attendees_emails: list, timezone: str = 'America/Chicago'):
        """
        Creates a Google Calendar event.
        """
        SCOPES = ['https://www.googleapis.com/auth/calendar']

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

    def _book_appointment(self, kwargs):
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
        result = self._create_google_calendar_meeting(
            f"Meeting for {patient_name}",
            dt_local.isoformat(),
            dt_local_end.isoformat(),
            ["altondsouza02@gmail.com", "aldsouza@ualberta.ca"]
        )
        
        # Extract the event link for better summary
        event_link = result.get('htmlLink', 'No link available') if result else 'No link available'
        
        return {
            "status": 200,
            "message": {
                "appointment_details": f"Appointment scheduled for {patient_name} with {doctor_specialty} on {date_str}",
                "event_link": event_link,
                "full_result": result
            }
        }

    def _check_heart_rate(self, kwargs):
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

    def _check_temperature(self, kwargs):
        """Check if body temperature is within normal range"""
        temperature = kwargs.get("temperature")
        
        print(f"Checking temperature: {temperature}°F")
        
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
                    "unit": "°F",
                    "status": temp_status,
                    "normal_range": f"{normal_temp_min}-{normal_temp_max}°F"
                },
                "assessment": f"Temperature is {temp_status}"
            }
        }

    def get_compiled_graph(self):
        """Get the compiled graph instance"""
        return self._compiled_graph

    def get_model(self):
        """Get the model instance"""
        return self._model

    def get_tokenizer(self):
        """Get the tokenizer instance"""
        return self._tokenizer

    def is_initialized(self):
        """Check if the agent is initialized"""
        return self._initialized


def get_health_agent():
    """
    Builder function to get the singleton instance of the health agent.
    This ensures only one instance is created and accessed throughout the application.

    Returns:
        HealthAgentSingleton: The singleton instance of the health agent
    """
    return HealthAgentSingleton()


# Example usage:
if __name__ == "__main__":
    # Get the singleton instance
    agent = get_health_agent()
    # Medical tools configuration
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
    messages = [
        {
            "content": system_prompt,
            "role": "system"
        },
        {
            "content": (
                "I have a headache and mild fever. What could be the possible conditions? "
                "Also, lookup medication details for 'Ibuprofen'. "
                "Please book an appointment for patient 'Alice Smith' with a neurologist on 2025-08-25. "
                "Check my heart rate - it's 85 bpm. Also check my temperature - it's 99.2°F."
            ),
            "role": "user"
        }
    ]
    # Access the compiled graph
    compiled_graph = agent.get_compiled_graph()
    compiled_graph.invoke({"messages":messages})

    # Access model and tokenizer if needed
    model = agent.get_model()
    tokenizer = agent.get_tokenizer()

    print("Agent singleton created successfully!")
    print(f"Agent initialized: {agent.is_initialized()}")

#!/usr/bin/env python3
"""
Dataset Generator for NAD Intelligence System
Generates training examples in conversation format for modern LLM training
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta


class NADDatasetGenerator:
    def __init__(self):
        # Medical AI Assistant tools definition - Simplified to 2 core functions
        self.tools = [
            {
                "name": "request_missing_argument",
                "description": "Ask the user for a required argument to complete a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "missing_field": {
                            "type": "string",
                            "description": "The specific field that is missing (e.g., 'patient_email', 'doctor_name', 'appointment_time', 'patient_id', 'appointment_id')"
                        }
                    },
                    "required": ["missing_field"]
                }
            },
            {
                "name": "book_appointment",
                "description": "Schedule an appointment with a doctor using doctor name and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doctor_name": {
                            "type": "string",
                            "description": "Full name of the doctor (e.g., 'Dr. Smith', 'Dr. Sarah Johnson')"
                        },
                        "appointment_time": {
                            "type": "string",
                            "description": "Time for appointment (e.g., '10:00 AM Monday', 'next available', 'ASAP', '2:30 PM tomorrow')"
                        }
                    },
                    "required": ["doctor_name", "appointment_time"]
                }
            }
        ]

        # Sample data for generating realistic examples
        self.doctors = [
            "Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Rodriguez", "Dr. David Kim",
            "Dr. Lisa Thompson", "Dr. Robert Wilson", "Dr. Amanda Patel", "Dr. James Lee",
            "Dr. Maria Garcia", "Dr. Thomas Brown", "Dr. Jennifer Davis", "Dr. Christopher Miller",
            "Dr. Rachel Green", "Dr. Alex Turner", "Dr. Sophia Martinez", "Dr. Daniel White",
            "Dr. Olivia Taylor", "Dr. Ethan Anderson", "Dr. Isabella Clark", "Dr. Noah Lewis",
            "Dr. Ava Hall", "Dr. William Young", "Dr. Mia King", "Dr. Lucas Scott",
            "Dr. Charlotte Adams", "Dr. Henry Baker", "Dr. Amelia Carter", "Dr. Sebastian Evans"
        ]

        self.specialties = [
            "dermatologist", "cardiologist", "pediatrician", "orthopedic surgeon",
            "neurologist", "psychiatrist", "endocrinologist", "gastroenterologist",
            "oncologist", "rheumatologist", "ophthalmologist", "urologist",
            "pulmonologist", "nephrologist", "hematologist", "infectious disease specialist",
            "allergist", "immunologist", "geriatrician", "obstetrician",
            "gynecologist", "plastic surgeon", "vascular surgeon", "neurosurgeon"
        ]

        self.time_formats = [
            "10:00 AM tomorrow", "2:30 PM Monday", "next available", "ASAP",
            "9:15 AM next Tuesday", "3:45 PM Friday", "next week", "this afternoon",
            "11:30 AM Wednesday", "4:00 PM Thursday", "first available slot", "next month",
            "8:00 AM next Monday", "1:15 PM Tuesday", "this weekend", "next available",
            "morning appointment", "afternoon slot", "evening appointment", "urgent care"
        ]

        self.date_ranges = [
            "next week", "this month", "next month", "this week",
            "Monday to Friday", "weekend", "next two weeks", "this quarter"
        ]

        self.medications = [
            "Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole",
            "Losartan", "Simvastatin", "Hydrochlorothiazide", "Atenolol", "Furosemide",
            "Warfarin", "Levothyroxine", "Albuterol", "Ibuprofen", "Acetaminophen",
            "Aspirin", "Vitamin D", "Calcium", "Iron", "Folic Acid"
        ]

        self.follow_up_types = [
            "post_surgery", "medication_review", "test_results", "routine_checkup",
            "specialist_consultation", "treatment_progress", "symptom_monitoring",
            "preventive_care", "chronic_condition_management", "recovery_assessment"
        ]

        self.medical_conditions = [
            "high blood pressure", "diabetes", "asthma", "arthritis", "depression",
            "anxiety", "heart disease", "thyroid disorder", "migraine", "back pain",
            "allergies", "insomnia", "obesity", "osteoporosis", "cancer"
        ]

        # Tasks that the assistant cannot perform
        self.unsupported_tasks = [
            "perform surgery", "prescribe medication", "diagnose medical conditions",
            "conduct medical procedures", "interpret X-rays", "perform blood tests",
            "provide emergency medical care", "make medical diagnoses", "perform physical exams",
            "administer treatments", "perform dental work", "conduct psychological therapy",
            "perform laboratory tests", "make treatment decisions", "provide legal advice",
            "process insurance claims", "handle billing disputes", "provide financial advice",
            "make travel arrangements", "order food delivery", "book hotel rooms",
            "schedule flights", "make restaurant reservations", "provide weather information",
            "give driving directions", "translate languages", "provide legal consultation"
        ]

        self.patient_ids = [f"user_{i:05d}" for i in range(1, 1001)]
        self.appointment_ids = [f"apt_{i:05d}" for i in range(1, 1001)]

        # User input variations
        self.booking_phrases = [
            "Book me with", "Schedule me with", "Make an appointment with", "I need to see",
            "Can you book", "I'd like to schedule", "Set up an appointment with", "I want to meet",
            "Please book", "I need an appointment with", "Can I get an appointment with", "I'd like to book"
        ]

        self.cancellation_phrases = [
            "Cancel my appointment", "I need to cancel", "Please cancel", "I can't make it",
            "I have to cancel", "Cancel the appointment", "I need to reschedule", "I can't attend",
            "Please reschedule", "I need to change", "I have to postpone", "Cancel my visit"
        ]

        self.update_phrases = [
            "I changed my", "My new", "Update my", "I need to update",
            "Please update", "Change my", "My contact info is now", "I moved to",
            "My new number is", "My new email is", "I got a new", "Please change my"
        ]

    def create_system_message(self) -> Dict[str, str]:
        """Create the system message with tools"""
        tools_json = json.dumps(self.tools, indent=2)
        system_content = f"""You are a medical AI assistant. Your job is to help users by understanding their request, identifying missing information if any, and then generating a structured plan using available tools.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, use the following JSON format within your response:
<tool_call>
{{"name": "function_name", "arguments": {{"param_name": "param_value"}}}}
</tool_call>

Available functions:
- request_missing_argument: Ask for missing information needed to complete a task
- book_appointment: Schedule appointments with doctors"""

        return {
            "role": "system",
            "content": system_content
        }

    def generate_basic_booking_example(self) -> Dict[str, Any]:
        """Generate example: Basic appointment booking with missing email"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        thoughts = f"The user has provided the essential information for booking an appointment: the doctor's name ({doctor}) and the desired appointment time ({time}). However, I notice that the patient's email address is missing, which is crucial for sending appointment confirmations and maintaining proper communication records. I should first request the missing email information to ensure we can provide proper confirmation and follow-up communications. Once the email is provided, I can proceed with the booking and generate a summary to confirm the appointment details."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll help you book an appointment with {doctor} at {time}. First, I need your email address to send you a confirmation.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My email is john.doe@email.com"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the user's email address. I can proceed with booking the appointment with {doctor} at {time}.\n</think>\nPerfect! I'll book your appointment now.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"{time}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_id\": \"apt_12345\", \"status\": \"confirmed\", \"confirmation_sent\": true}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe appointment has been successfully booked and confirmed. I should provide a clear summary of what was accomplished and next steps for the user.\n</think>\nGreat! Your appointment has been successfully booked. Here's a summary of what I've done:\n\n✅ **Appointment Confirmed**: {doctor} at {time}\n✅ **Confirmation Email**: Sent to john.doe@email.com\n✅ **Appointment ID**: apt_12345\n\nYou'll receive a confirmation email shortly with all the details. Please arrive 10 minutes before your scheduled time. If you need to make any changes, just let me know!"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_incremental_booking_example(self) -> Dict[str, Any]:
        """Generate example: Incremental booking with step-by-step process"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        thoughts_step1 = f"The user wants to book an appointment with {doctor} at {time}. I need to gather all required information step by step. First, I should ask for their email address to ensure proper communication."

        thoughts_step2 = f"Now I have their email. Next, I should ask for their phone number for emergency contact purposes."

        thoughts_step3 = f"Great! Now I have all the contact information. I should ask for their preferred appointment duration to ensure proper scheduling."

        thoughts_step4 = f"Perfect! Now I have all the necessary information. I can proceed with booking the appointment."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step1}\n</think>\nI'll help you book an appointment with {doctor} at {time}. Let me gather the necessary information step by step.\n\nFirst, I need your email address for confirmation:\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My email is john.doe@email.com"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step2}\n</think>\nThank you! Now I need your phone number for emergency contact:\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_phone\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My phone is 555-123-4567"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step3}\n</think>\nPerfect! Now, what's your preferred appointment duration?\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_duration\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "30 minutes should be fine"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step4}\n</think>\nExcellent! Now I have all the information. Let me book your appointment:\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"{time}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_id\": \"apt_12345\", \"status\": \"confirmed\", \"confirmation_sent\": true, \"duration\": \"30 minutes\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe appointment has been successfully booked with all the gathered information. I should provide a comprehensive summary.\n</think>\nPerfect! Your appointment has been successfully booked. Here's a summary:\n\n✅ **Appointment Confirmed**: {doctor} at {time}\n✅ **Duration**: 30 minutes\n✅ **Email**: john.doe@email.com\n✅ **Phone**: 555-123-4567\n✅ **Appointment ID**: apt_12345\n\nYou'll receive a confirmation email shortly. Please arrive 10 minutes early!"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_incremental_cancellation_example(self) -> Dict[str, Any]:
        """Generate example: Incremental cancellation with verification steps"""
        doctor = random.choice(self.doctors)
        appointment_id = random.choice(self.appointment_ids)
        phrase = random.choice(self.cancellation_phrases)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = random.choice(days)
        user_input = f"{phrase} with {doctor} scheduled for {day}."

        thoughts_step1 = f"The user wants to cancel their appointment with {doctor} on {day}. I need to gather information step by step to ensure accuracy. First, I should ask for the appointment ID."

        thoughts_step2 = f"Now I have the appointment ID. I should verify the appointment details before proceeding with cancellation."

        thoughts_step3 = f"The appointment details are confirmed. Now I should ask for the reason for cancellation to help improve our services."

        thoughts_step4 = f"Perfect! Now I have all the information needed. I can proceed with the cancellation."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step1}\n</think>\nI'll help you cancel your appointment with {doctor} on {day}. Let me gather the necessary information step by step.\n\nFirst, I need your appointment ID:\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My appointment ID is {appointment_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step2}\n</think>\nThank you! Let me verify your appointment details:\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"verify\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_found\": true, \"doctor\": \"" + doctor + "\", \"day\": \"" + day + "\", \"time\": \"2:00 PM\", \"duration\": \"30 minutes\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step3}\n</think>\nPerfect! I found your appointment with {doctor} on {day} at 2:00 PM. Now, what's the reason for cancellation? This helps us improve our services:\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"cancellation_reason\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "I have a conflict with another appointment"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts_step4}\n</think>\nThank you for letting us know. Now I'll proceed with the cancellation:\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"cancelled\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"cancelled\", \"appointment_id\": \"" + appointment_id + "\", \"cancellation_confirmed\": true, \"refund_processed\": true, \"reason\": \"conflict with another appointment\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe appointment has been successfully cancelled with all the gathered information. I should provide a comprehensive summary.\n</think>\nPerfect! I've successfully cancelled your appointment. Here's a summary:\n\n✅ **Appointment Cancelled**: {doctor} on {day} at 2:00 PM\n✅ **Appointment ID**: {appointment_id}\n✅ **Reason**: Conflict with another appointment\n✅ **Status**: Cancellation confirmed\n✅ **Refund**: Processed automatically\n\nYou'll receive a cancellation confirmation email shortly. Would you like me to help you reschedule with {doctor} or book with a different doctor?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_patient_info_update_example(self) -> Dict[str, Any]:
        """Generate example: Patient information update"""
        patient_id = random.choice(self.patient_ids)
        fields = ["phone_number", "email", "address", "emergency_contact"]
        field = random.choice(fields)
        phrase = random.choice(self.update_phrases)

        if field == "phone_number":
            value = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif field == "email":
            domains = ["gmail.com", "yahoo.com", "outlook.com", "healthcare.com", "patient.com"]
            value = f"patient{random.randint(1, 999)}@{random.choice(domains)}"
        elif field == "address":
            streets = ["Main St", "Oak Ave", "Pine Rd", "Elm St", "Cedar Ln", "Maple Dr"]
            value = f"{random.randint(100, 9999)} {random.choice(streets)}"
        else:
            value = f"Emergency Contact {random.randint(1, 999)}"

        user_input = f"{phrase} {field.replace('_', ' ')} to {value}."

        thoughts = f"The user is requesting to update their {field.replace('_', ' ')} to {value}. This is an important administrative task that requires updating their contact information in our patient database. However, I need to identify which specific patient record to update, as the user hasn't provided their patient ID. This is crucial for data accuracy and security - we need to ensure we're updating the correct patient's information. I should request the patient ID first, then proceed with updating the specified field with the new value, and finally provide a confirmation that the information has been successfully updated."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll help you update your {field.replace('_', ' ')}. To ensure I update the correct record, I need your patient ID.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_medical_records_example(self) -> Dict[str, Any]:
        """Generate example: Medical records request"""
        patient_id = random.choice(self.patient_ids)
        record_variations = [
            "Can you show me my past reports?",
            "I need to see my medical history.",
            "Please fetch my medical records.",
            "Can I get my test results?",
            "I want to review my previous appointments.",
            "Show me my medical history before booking."
        ]
        user_input = random.choice(record_variations)

        thoughts = "The user is requesting access to their medical records, which may include past reports, test results, and appointment history. This is a sensitive request that requires proper patient identification to ensure we're providing the correct medical information to the right person. Medical records contain confidential health information, so I need to verify the patient's identity by requesting their patient ID before proceeding. Once I have the proper identification, I can securely retrieve their medical records and provide them with access to their health information. This helps patients stay informed about their health status and make informed decisions about their care."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": "<think>\n{thoughts}\n</think>\nI'll help you access your medical records. For security purposes, I need to verify your identity with your patient ID.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can securely retrieve their medical records.\n</think>\nThank you! I'll retrieve your medical records now.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"Medical Records\", \"appointment_time\": \"now\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"records_found\": 15, \"last_visit\": \"2024-01-15\", \"recent_tests\": [\"Blood Test\", \"X-Ray\"], \"upcoming_appointments\": 2, \"medications\": [\"Metformin\", \"Lisinopril\"]}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe medical records have been successfully retrieved. I should provide a comprehensive summary of what was found and help the user understand their medical information.\n</think>\nPerfect! I've successfully retrieved your medical records. Here's a summary of what I found:\n\n📋 **Medical Records Summary**:\n✅ **Total Records**: 15 medical records found\n✅ **Last Visit**: January 15, 2024\n✅ **Recent Tests**: Blood Test, X-Ray\n✅ **Upcoming Appointments**: 2 scheduled\n✅ **Current Medications**: Metformin, Lisinopril\n\nYour medical history is now available for review. You can access detailed reports, test results, and appointment history through your patient portal. If you need help understanding any of your medical information or would like to discuss your records with a doctor, just let me know!"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_medical_info_query_example(self) -> Dict[str, Any]:
        """Generate example: Medical information query"""
        condition = random.choice(self.medical_conditions)
        doctor = random.choice(self.doctors)

        query_variations = [
            f"Does {doctor} treat {condition}?",
            f"Can {doctor} handle {condition} cases?",
            f"Is {doctor} experienced with {condition}?",
            f"Does {doctor} specialize in {condition}?",
            f"Can you tell me if {doctor} treats {condition}?"
        ]
        user_input = random.choice(query_variations)

        thoughts = f"The user is seeking information about whether {doctor} has expertise in treating {condition}. This is an important query that helps patients make informed decisions about their healthcare provider selection. The user wants to ensure that the doctor they're considering has the appropriate experience and specialization for their specific medical condition. I should search our medical knowledge base to find information about {doctor}'s qualifications, experience, and specialization in {condition} treatment. This will help the user determine if this doctor is the right fit for their healthcare needs and provide them with confidence in their choice of healthcare provider."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll search for information about {doctor}'s expertise in treating {condition}.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"consultation\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"doctor\": \"" + doctor + "\", \"specialty\": \"Cardiologist\", \"experience_years\": 12, \"treats_condition\": true, \"success_rate\": \"95%\", \"patient_reviews\": \"Excellent\", \"certifications\": [\"Board Certified Cardiologist\", \"Advanced Heart Failure Specialist\"]}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI found comprehensive information about {doctor}'s expertise in treating {condition}. I should provide a clear summary of the findings to help the user make an informed decision.\n</think>\nGreat news! I found detailed information about {doctor}'s expertise in treating {condition}. Here's what I discovered:\n\n👨‍⚕️ **Doctor Information**:\n✅ **Name**: {doctor}\n✅ **Specialty**: Cardiologist\n✅ **Experience**: 12 years\n✅ **Treats {condition}**: Yes\n✅ **Success Rate**: 95%\n✅ **Patient Reviews**: Excellent\n✅ **Certifications**: Board Certified Cardiologist, Advanced Heart Failure Specialist\n\n**Summary**: {doctor} is highly qualified to treat {condition} with extensive experience and excellent patient outcomes. You can feel confident in choosing {doctor} for your {condition} treatment. Would you like me to help you book an appointment with {doctor}?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_doctor_profile_example(self) -> Dict[str, Any]:
        """Generate example: Doctor profile request"""
        doctor = random.choice(self.doctors)

        profile_variations = [
            f"Tell me about {doctor}.",
            f"What's {doctor}'s background?",
            f"Can I get {doctor}'s profile?",
            f"Tell me about {doctor}'s experience.",
            f"I want to know more about {doctor}."
        ]
        user_input = random.choice(profile_variations)

        thoughts = f"The user is requesting detailed information about {doctor}, which suggests they want to learn more about the doctor's qualifications, experience, education, and background before making a decision about their healthcare. This is a common and important step in the patient-doctor relationship building process. Patients often want to feel confident about their healthcare provider's credentials and experience. I should retrieve {doctor}'s comprehensive profile, which typically includes their medical education, board certifications, years of experience, areas of expertise, and any specializations. This information helps patients make informed decisions and builds trust in the healthcare relationship."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll retrieve {doctor}'s profile and background information for you.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"profile_review\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"name\": \"" + doctor + "\", \"specialty\": \"Internal Medicine\", \"education\": \"Harvard Medical School\", \"experience_years\": 15, \"board_certifications\": [\"American Board of Internal Medicine\"], \"languages\": [\"English\", \"Spanish\"], \"patient_satisfaction\": \"98%\", \"awards\": [\"Top Doctor 2023\", \"Patient Choice Award\"]}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI retrieved comprehensive information about {doctor}'s background and qualifications. I should provide a clear summary of the doctor's profile to help the user make an informed decision.\n</think>\nHere's {doctor}'s complete profile and background:\n\n👨‍⚕️ **Doctor Profile**:\n✅ **Name**: {doctor}\n✅ **Specialty**: Internal Medicine\n✅ **Education**: Harvard Medical School\n✅ **Experience**: 15 years\n✅ **Board Certifications**: American Board of Internal Medicine\n✅ **Languages**: English, Spanish\n✅ **Patient Satisfaction**: 98%\n✅ **Awards**: Top Doctor 2023, Patient Choice Award\n\n**Summary**: {doctor} is a highly qualified and experienced internal medicine specialist with excellent patient satisfaction ratings. With 15 years of experience and board certification, {doctor} provides comprehensive primary care services. The ability to speak both English and Spanish makes {doctor} accessible to a diverse patient population. The recent awards indicate recognition for exceptional patient care and medical expertise."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_availability_check_example(self) -> Dict[str, Any]:
        """Generate example: Check doctor availability"""
        doctor = random.choice(self.doctors)
        date_range = random.choice(self.date_ranges)

        availability_variations = [
            f"Is {doctor} available {date_range}?",
            f"Check {doctor}'s availability {date_range}.",
            f"When is {doctor} free {date_range}?",
            f"Can you check if {doctor} has openings {date_range}?",
            f"What's {doctor}'s schedule {date_range}?"
        ]
        user_input = random.choice(availability_variations)

        thoughts = f"The user is inquiring about {doctor}'s availability during {date_range}, which indicates they're interested in scheduling an appointment with this specific doctor. This is a common step in the appointment booking process where patients want to know when their preferred doctor has open slots before committing to a specific time. I should check the doctor's schedule for the specified time period to identify available appointment slots. This information helps patients plan their appointments around their own schedules and preferences, ensuring they can see the doctor they want at a convenient time. Providing accurate availability information is crucial for patient satisfaction and efficient scheduling."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll check {doctor}'s availability for {date_range}.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"{date_range}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_prescription_refill_example(self) -> Dict[str, Any]:
        """Generate example: Prescription refill request"""
        medication = random.choice(self.medications)
        patient_id = random.choice(self.patient_ids)

        refill_variations = [
            f"I need a refill for {medication}.",
            f"Can you refill my {medication} prescription?",
            f"I'm running low on {medication}.",
            f"Please refill {medication}.",
            f"I need more {medication}."
        ]
        user_input = random.choice(refill_variations)

        thoughts = f"The user is requesting a prescription refill for {medication}, which indicates they're running low on their medication and need to continue their treatment regimen. This is an important healthcare request that requires proper patient identification to ensure we're processing the refill for the correct patient and medication. Prescription refills are critical for maintaining continuity of care and ensuring patients don't experience gaps in their medication therapy. I need to first identify the patient by requesting their patient ID, then create a refill request that will be sent to their prescribing doctor for approval. This process ensures medication safety and proper medical oversight while providing convenient access to necessary medications."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll help you request a refill for {medication}. To process this correctly, I need your patient ID.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_follow_up_example(self) -> Dict[str, Any]:
        """Generate example: Follow-up appointment scheduling"""
        appointment_id = random.choice(self.appointment_ids)
        follow_up_type = random.choice(self.follow_up_types)

        follow_up_variations = [
            f"I need a follow-up appointment.",
            f"Schedule a follow-up for me.",
            f"I need to come back for a follow-up.",
            f"Can you set up a follow-up appointment?",
            f"I need a follow-up visit."
        ]
        user_input = random.choice(follow_up_variations)

        thoughts = "The user is requesting a follow-up appointment, which is an important part of ongoing healthcare management. Follow-up appointments are typically scheduled after initial consultations, treatments, or procedures to monitor progress, review test results, or continue treatment plans. To properly schedule the follow-up, I need to identify which current appointment this follow-up is related to, as this helps determine the appropriate timing and type of follow-up needed. Different types of follow-ups (such as post-surgery, medication review, or test results) may have different scheduling requirements and intervals. I should request the appointment ID to link this follow-up to the correct previous appointment, then schedule the appropriate type of follow-up based on the context of their care."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": "<think>\n{thoughts}\n</think>\nI'll help you schedule a follow-up appointment. To link it to your current care, I need your appointment ID.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_complex_booking_example(self) -> Dict[str, Any]:
        """Generate example: Complex booking with multiple requirements"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        condition = random.choice(self.medical_conditions)

        complex_variations = [
            f"Book me with {doctor} {time} and also get my medical records.",
            f"I need to see {doctor} {time} and check my test results.",
            f"Schedule {doctor} {time} and show me my history.",
            f"Book {doctor} {time} and fetch my records."
        ]
        user_input = random.choice(complex_variations)

        thoughts = f"The user is making a multi-part request that involves both scheduling an appointment with {doctor} at {time} and retrieving their medical records. This is a comprehensive healthcare request that suggests the user wants to be well-prepared for their appointment by having access to their medical history. This type of request is common when patients want to review their health information before meeting with their doctor, ensuring they can have informed discussions about their care. I need to handle both aspects of this request: first, I should request the missing email for appointment confirmation, then proceed with the booking. Additionally, I need to request the patient ID to securely access their medical records. This approach ensures the user gets both the appointment they need and the medical information they want to review."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI'll help you book an appointment with {doctor} and retrieve your medical records. First, I need your email for the appointment confirmation.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_task_example(self) -> Dict[str, Any]:
        """Generate example: Tasks the assistant cannot perform"""
        unsupported_task = random.choice(self.unsupported_tasks)

        task_variations = [
            f"Can you {unsupported_task}?",
            f"I need you to {unsupported_task}.",
            f"Please {unsupported_task} for me.",
            f"Help me {unsupported_task}.",
            f"I want you to {unsupported_task}.",
            f"Can you help me {unsupported_task}?",
            f"I need help with {unsupported_task}.",
            f"Please assist me with {unsupported_task}."
        ]
        user_input = random.choice(task_variations)

        thoughts = f"The user is requesting me to {unsupported_task}, which is outside of my capabilities as a medical appointment booking assistant. I should politely explain that I cannot perform this task and clarify what I can help them with instead. It's important to be honest about my limitations while still being helpful and professional."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI apologize, but I don't have the ability to {unsupported_task}. As a medical appointment booking assistant, my capabilities are focused on helping you with appointment scheduling, medical records access, doctor information, and administrative tasks related to your healthcare.\n\nI can help you with:\n✅ Booking and managing appointments\n✅ Finding doctors by specialty\n✅ Accessing your medical records\n✅ Updating your contact information\n✅ Checking doctor availability\n✅ Prescription refill requests\n✅ Follow-up appointment scheduling\n\nIs there something related to appointment booking or medical administration that I can help you with instead?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_medical_diagnosis_request_example(self) -> Dict[str, Any]:
        """Generate example: Medical diagnosis requests (not supported)"""
        symptoms = [
            "I have chest pain", "I'm experiencing headaches", "I feel dizzy",
            "I have a fever", "My stomach hurts", "I'm having trouble breathing",
            "I feel nauseous", "I have joint pain", "I'm feeling very tired",
            "I have a rash", "My vision is blurry", "I have trouble sleeping"
        ]
        symptom = random.choice(symptoms)

        diagnosis_variations = [
            f"{symptom}. What's wrong with me?",
            f"{symptom}. Can you diagnose me?",
            f"{symptom}. What do you think I have?",
            f"{symptom}. What's my diagnosis?",
            f"{symptom}. Can you tell me what's wrong?",
            f"{symptom}. What should I do?",
            f"{symptom}. Is this serious?",
            f"{symptom}. What could this be?"
        ]
        user_input = random.choice(diagnosis_variations)

        thoughts = f"The user is describing symptoms and asking for a medical diagnosis, which is outside of my capabilities as a medical appointment booking assistant. I cannot and should not provide medical diagnoses, as this requires proper medical training and examination. I should explain my limitations, express concern for their health, and encourage them to seek proper medical care."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI understand you're concerned about your symptoms, but I don't have the ability to provide medical diagnoses. As a medical appointment booking assistant, I can help you schedule appointments with qualified healthcare providers who can properly evaluate your symptoms.\n\nFor your symptoms, I recommend:\n🏥 **Seek Medical Attention**: Contact your primary care physician or visit an urgent care facility\n🏥 **Emergency Care**: If symptoms are severe, consider going to the emergency room\n🏥 **Book an Appointment**: I can help you schedule an appointment with a doctor who can properly diagnose your condition\n\nWould you like me to help you book an appointment with a doctor who can evaluate your symptoms? I can find available appointments with appropriate specialists."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_emergency_request_example(self) -> Dict[str, Any]:
        """Generate example: Emergency medical requests (not supported)"""
        emergency_situations = [
            "I'm having a heart attack", "I can't breathe", "I'm bleeding heavily",
            "I think I broke my arm", "I'm having severe chest pain", "I lost consciousness",
            "I'm having a seizure", "I have severe abdominal pain", "I'm having trouble speaking",
            "I have a severe allergic reaction", "I think I have a stroke", "I'm in severe pain"
        ]
        emergency = random.choice(emergency_situations)

        emergency_variations = [
            f"{emergency}. Help me!",
            f"{emergency}. What should I do?",
            f"{emergency}. Can you help?",
            f"{emergency}. I need immediate assistance!",
            f"{emergency}. This is an emergency!",
            f"{emergency}. Please help me now!"
        ]
        user_input = random.choice(emergency_variations)

        thoughts = f"The user is describing what appears to be a medical emergency. I cannot provide emergency medical care or advice, and I should immediately direct them to call emergency services. This is a critical situation where the user needs immediate professional medical attention."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\n🚨 **EMERGENCY SITUATION** 🚨\n\nI cannot provide emergency medical care or advice. This sounds like a serious medical emergency that requires immediate professional attention.\n\n**Please take action immediately:**\n\n📞 **Call Emergency Services**: Dial 911 (or your local emergency number)\n🏥 **Go to Emergency Room**: If possible, go to the nearest emergency room\n🚑 **Call Ambulance**: If you cannot safely transport yourself\n\n**Do not delay** - your health and safety are the top priority. Emergency medical professionals are trained to handle these situations and can provide the immediate care you need.\n\nOnce you've received emergency care, I can help you with follow-up appointments and medical record management."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_prescription_request_example(self) -> Dict[str, Any]:
        """Generate example: Prescription requests (not supported)"""
        medications = [
            "antibiotics", "pain medication", "sleeping pills", "anxiety medication",
            "blood pressure medication", "diabetes medication", "cholesterol medication",
            "birth control", "antidepressants", "ADHD medication", "migraine medication"
        ]
        medication = random.choice(medications)

        prescription_variations = [
            f"I need a prescription for {medication}.",
            f"Can you prescribe {medication}?",
            f"I want {medication}. Can you give me a prescription?",
            f"Please prescribe {medication} for me.",
            f"I need {medication}. Can you write me a prescription?",
            f"Can you give me a prescription for {medication}?"
        ]
        user_input = random.choice(prescription_variations)

        thoughts = f"The user is asking me to prescribe medication, which is outside of my capabilities. I cannot prescribe medications as I'm not a licensed healthcare provider. I should explain this limitation and suggest they consult with a qualified doctor who can properly evaluate their needs and prescribe appropriate medications."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{thoughts}\n</think>\nI understand you need {medication}, but I don't have the ability to prescribe medications. As a medical appointment booking assistant, I cannot write prescriptions or make medical decisions about medications.\n\n**What I can help you with:**\n✅ **Book an appointment** with a doctor who can prescribe {medication}\n✅ **Request prescription refills** for existing medications\n✅ **Find specialists** who can evaluate your medication needs\n✅ **Schedule follow-up appointments** for medication management\n\n**Recommendation:**\nPlease schedule an appointment with a qualified healthcare provider who can:\n🏥 Evaluate your medical condition\n🏥 Determine if {medication} is appropriate for you\n🏥 Write a proper prescription if needed\n🏥 Monitor your response to the medication\n\nWould you like me to help you book an appointment with a doctor who can assist with your medication needs?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_dataset(self, num_examples: int = 200) -> List[Dict[str, Any]]:
        """Generate a complete dataset with various examples"""
        dataset = []

        # Define example generators and their weights - Focus on incremental conversations
        generators = [
            (self.generate_incremental_booking_example, 0.40),
            (self.generate_incremental_cancellation_example, 0.30),
            (self.generate_basic_booking_example, 0.20),
            (self.generate_medical_records_example, 0.10)
        ]

        for _ in range(num_examples):
            # Select generator based on weights
            generator, weight = random.choices(generators, weights=[w for _, w in generators])[0]
            example = generator()
            dataset.append(example)

        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "nad_training_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename} with {len(dataset)} examples")

    def print_example(self, example: Dict[str, Any]):
        """Print a formatted example"""
        print("=" * 80)
        print("CONVERSATION EXAMPLE:")
        print("=" * 80)

        for message in example['messages']:
            if message['role'] == 'system':
                print(f"SYSTEM: [System message with {len(example['tools'])} tools]")
            elif message['role'] == 'user':
                print(f"USER: {message['content']}")
            elif message['role'] == 'assistant':
                print(f"ASSISTANT: {message['content']}")

        print("=" * 80)

    def generate_statistics(self, dataset: List[Dict[str, Any]]):
        """Generate and print dataset statistics"""
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)

        # Count tool usage
        tool_counts = {}
        total_tool_calls = 0

        for example in dataset:
            for message in example['messages']:
                if message['role'] == 'assistant' and '<tool_call>' in message['content']:
                    # Count tool calls in the content
                    content = message['content']
                    if '"name": "request_missing_argument"' in content:
                        tool_counts['request_missing_argument'] = tool_counts.get('request_missing_argument', 0) + 1
                        total_tool_calls += 1
                    if '"name": "book_appointment"' in content:
                        tool_counts['book_appointment'] = tool_counts.get('book_appointment', 0) + 1
                        total_tool_calls += 1

        print(f"Total examples: {len(dataset)}")
        print(f"Total tool calls: {total_tool_calls}")
        print(f"Unique tools used: {len(tool_counts)}")
        print("\nTool usage:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} times")

        # Average tool calls per example
        avg_tool_calls = total_tool_calls / len(dataset) if dataset else 0
        print(f"\nAverage tool calls per example: {avg_tool_calls:.2f}")

        # User input length statistics
        input_lengths = []
        for example in dataset:
            for message in example['messages']:
                if message['role'] == 'user':
                    input_lengths.append(len(message['content']))

        if input_lengths:
            avg_input_length = sum(input_lengths) / len(input_lengths)
            print(f"Average user input length: {avg_input_length:.1f} characters")


def main():
    """Main function to generate and save dataset"""
    generator = NADDatasetGenerator()

    # Generate dataset
    print("Generating NAD Intelligence training dataset in conversation format...")
    train_dataset = generator.generate_dataset(num_examples=200)
    test_dataset = generator.generate_dataset(num_examples=30)

    # Save dataset
    generator.save_dataset(train_dataset, "nad_training_dataset.json")
    generator.save_dataset(test_dataset, "nad_test_dataset.json")

    # Print statistics
    generator.generate_statistics(train_dataset)
    generator.generate_statistics(test_dataset)

    # Print a few examples
    print("\nSample examples:")
    for i in range(min(3, len(train_dataset))):
        generator.print_example(train_dataset[i])

    print(f"\nDataset generation complete! Generated {len(train_dataset)} training examples in conversation format.")


if __name__ == "__main__":
    main()
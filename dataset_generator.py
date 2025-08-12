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
        # Comprehensive AI Assistant tools definition - Medical + General functions
        self.tools = [
            {
                "name": "request_missing_argument",
                "description": "Ask the user for a required argument to complete a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "missing_field": {
                            "type": "string",
                            "description": "The specific field that is missing (e.g., 'patient_email', 'doctor_name', 'appointment_time', 'patient_id', 'appointment_id', 'email', 'phone', 'address', 'name')"
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
            },
            {
                "name": "cancel_appointment",
                "description": "Cancel an existing appointment using appointment ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {
                            "type": "string",
                            "description": "The appointment ID to cancel"
                        }
                    },
                    "required": ["appointment_id"]
                }
            },
            {
                "name": "reschedule_appointment",
                "description": "Reschedule an existing appointment to a new time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {
                            "type": "string",
                            "description": "The appointment ID to reschedule"
                        },
                        "new_time": {
                            "type": "string",
                            "description": "The new appointment time"
                        }
                    },
                    "required": ["appointment_id", "new_time"]
                }
            },
            {
                "name": "get_medical_records",
                "description": "Retrieve patient medical records using patient ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID to retrieve records for"
                        }
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "update_patient_info",
                "description": "Update patient contact information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID to update"
                        },
                        "field": {
                            "type": "string",
                            "description": "The field to update (e.g., 'email', 'phone', 'address')"
                        },
                        "new_value": {
                            "type": "string",
                            "description": "The new value for the field"
                        }
                    },
                    "required": ["patient_id", "field", "new_value"]
                }
            },
            {
                "name": "find_doctor",
                "description": "Find doctors by specialty or availability",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "specialty": {
                            "type": "string",
                            "description": "Medical specialty (e.g., 'cardiologist', 'dermatologist')"
                        },
                        "availability": {
                            "type": "string",
                            "description": "Preferred availability (e.g., 'next week', 'ASAP')"
                        }
                    },
                    "required": ["specialty"]
                }
            },
            {
                "name": "request_prescription_refill",
                "description": "Request a prescription refill for existing medication",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        },
                        "medication": {
                            "type": "string",
                            "description": "Name of the medication to refill"
                        }
                    },
                    "required": ["patient_id", "medication"]
                }
            },
            {
                "name": "get_test_results",
                "description": "Retrieve patient test results using patient ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID to retrieve test results for"
                        },
                        "test_type": {
                            "type": "string",
                            "description": "Type of test results to retrieve (e.g., 'blood test', 'x-ray', 'mri', 'all')"
                        }
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "schedule_lab_test",
                "description": "Schedule a laboratory test for a patient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        },
                        "test_type": {
                            "type": "string",
                            "description": "Type of test to schedule (e.g., 'blood test', 'urine test', 'x-ray', 'mri')"
                        },
                        "preferred_time": {
                            "type": "string",
                            "description": "Preferred time for the test"
                        }
                    },
                    "required": ["patient_id", "test_type"]
                }
            },
            {
                "name": "get_doctor_schedule",
                "description": "Get a doctor's schedule and availability",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doctor_name": {
                            "type": "string",
                            "description": "Full name of the doctor"
                        },
                        "date_range": {
                            "type": "string",
                            "description": "Date range to check (e.g., 'next week', 'this month')"
                        }
                    },
                    "required": ["doctor_name"]
                }
            },
            {
                "name": "get_patient_appointments",
                "description": "Get all appointments for a specific patient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        },
                        "date_range": {
                            "type": "string",
                            "description": "Date range to check (e.g., 'upcoming', 'past month', 'all')"
                        }
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "send_reminder",
                "description": "Send appointment reminder to patient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {
                            "type": "string",
                            "description": "The appointment ID"
                        },
                        "reminder_type": {
                            "type": "string",
                            "description": "Type of reminder (e.g., 'email', 'sms', 'both')"
                        }
                    },
                    "required": ["appointment_id"]
                }
            },
            {
                "name": "get_insurance_info",
                "description": "Get patient insurance information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        }
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "update_insurance_info",
                "description": "Update patient insurance information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        },
                        "insurance_provider": {
                            "type": "string",
                            "description": "Insurance provider name"
                        },
                        "policy_number": {
                            "type": "string",
                            "description": "Insurance policy number"
                        }
                    },
                    "required": ["patient_id", "insurance_provider", "policy_number"]
                }
            },
            {
                "name": "get_medication_list",
                "description": "Get patient's current medication list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        }
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "add_medication",
                "description": "Add medication to patient's list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        },
                        "medication_name": {
                            "type": "string",
                            "description": "Name of the medication"
                        },
                        "dosage": {
                            "type": "string",
                            "description": "Dosage information"
                        },
                        "frequency": {
                            "type": "string",
                            "description": "How often to take the medication"
                        }
                    },
                    "required": ["patient_id", "medication_name", "dosage", "frequency"]
                }
            },
            {
                "name": "get_allergies",
                "description": "Get patient's allergy information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        }
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "add_allergy",
                "description": "Add allergy to patient's record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The patient ID"
                        },
                        "allergy_name": {
                            "type": "string",
                            "description": "Name of the allergy"
                        },
                        "severity": {
                            "type": "string",
                            "description": "Severity level (e.g., 'mild', 'moderate', 'severe')"
                        }
                    },
                    "required": ["patient_id", "allergy_name", "severity"]
                }
            },
            {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "search_type": {
                            "type": "string",
                            "description": "Type of search (e.g., 'general', 'news', 'images', 'videos')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            },
            {
                "name": "create_calendar_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Event title"
                        },
                        "date": {
                            "type": "string",
                            "description": "Event date"
                        },
                        "time": {
                            "type": "string",
                            "description": "Event time"
                        },
                        "duration": {
                            "type": "string",
                            "description": "Event duration"
                        }
                    },
                    "required": ["title", "date", "time"]
                }
            },
            {
                "name": "set_reminder",
                "description": "Set a reminder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task to remind about"
                        },
                        "date": {
                            "type": "string",
                            "description": "Reminder date"
                        },
                        "time": {
                            "type": "string",
                            "description": "Reminder time"
                        }
                    },
                    "required": ["task", "date", "time"]
                }
            },
            {
                "name": "translate_text",
                "description": "Translate text to another language",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to translate"
                        },
                        "target_language": {
                            "type": "string",
                            "description": "Target language"
                        }
                    },
                    "required": ["text", "target_language"]
                }
            },
            {
                "name": "calculate_math",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to calculate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location for weather information"
                        },
                        "date": {
                            "type": "string",
                            "description": "Date for weather forecast (optional)"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "book_travel",
                "description": "Book travel arrangements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "Travel destination"
                        },
                        "date": {
                            "type": "string",
                            "description": "Travel date"
                        },
                        "travel_type": {
                            "type": "string",
                            "description": "Type of travel (flight, hotel, car rental)"
                        }
                    },
                    "required": ["destination", "date", "travel_type"]
                }
            },
            {
                "name": "order_food",
                "description": "Order food from a restaurant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restaurant": {
                            "type": "string",
                            "description": "Restaurant name"
                        },
                        "items": {
                            "type": "string",
                            "description": "Food items to order"
                        },
                        "delivery_address": {
                            "type": "string",
                            "description": "Delivery address"
                        }
                    },
                    "required": ["restaurant", "items"]
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

        # New data for additional tools
        self.test_types = [
            "blood test", "urine test", "x-ray", "mri", "ct scan", "ultrasound",
            "ekg", "stress test", "mammogram", "colonoscopy", "endoscopy",
            "biopsy", "pregnancy test", "covid test", "flu test", "strep test"
        ]

        self.insurance_providers = [
            "Blue Cross Blue Shield", "Aetna", "Cigna", "UnitedHealth", "Kaiser Permanente",
            "Humana", "Anthem", "Molina Healthcare", "Medicare", "Medicaid",
            "AARP", "Health Net", "Oscar Health", "Bright Health", "Ambetter"
        ]

        self.allergies = [
            "penicillin", "aspirin", "ibuprofen", "sulfa drugs", "codeine",
            "morphine", "latex", "eggs", "peanuts", "tree nuts", "milk",
            "soy", "wheat", "fish", "shellfish", "dust", "pollen", "mold",
            "pet dander", "bee stings", "latex", "iodine", "tetracycline"
        ]

        self.allergy_severities = ["mild", "moderate", "severe"]

        self.medication_frequencies = [
            "once daily", "twice daily", "three times daily", "four times daily",
            "every 6 hours", "every 8 hours", "every 12 hours", "as needed",
            "before meals", "after meals", "at bedtime", "in the morning"
        ]

        self.dosages = [
            "10mg", "20mg", "50mg", "100mg", "250mg", "500mg", "1g",
            "5ml", "10ml", "15ml", "1 tablet", "2 tablets", "1 capsule",
            "1 teaspoon", "1 tablespoon", "1 drop", "2 drops"
        ]

        # General assistant data
        self.languages = [
            "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese", "Korean",
            "Arabic", "Russian", "Hindi", "Dutch", "Swedish", "Norwegian", "Danish", "Finnish"
        ]

        self.cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
            "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus",
            "Charlotte", "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington DC"
        ]

        self.restaurants = [
            "McDonald's", "Burger King", "KFC", "Subway", "Pizza Hut", "Domino's", "Taco Bell",
            "Wendy's", "Starbucks", "Dunkin'", "Chipotle", "Panera Bread", "Chick-fil-A",
            "Popeyes", "Arby's", "Sonic", "Five Guys", "Shake Shack", "In-N-Out", "Whataburger"
        ]

        self.food_items = [
            "hamburger", "chicken sandwich", "pizza", "tacos", "salad", "fries", "nuggets",
            "burrito", "sandwich", "coffee", "milkshake", "ice cream", "soup", "pasta",
            "steak", "fish", "rice", "noodles", "bread", "dessert"
        ]

        self.travel_destinations = [
            "Paris", "London", "Tokyo", "New York", "Rome", "Barcelona", "Amsterdam", "Berlin",
            "Prague", "Vienna", "Budapest", "Istanbul", "Dubai", "Singapore", "Hong Kong",
            "Sydney", "Melbourne", "Toronto", "Vancouver", "Montreal"
        ]

        self.travel_types = [
            "flight", "hotel", "car rental", "train", "bus", "cruise", "package deal"
        ]

        self.event_types = [
            "meeting", "appointment", "birthday party", "dinner", "conference", "workshop",
            "doctor visit", "dentist appointment", "haircut", "massage", "spa treatment"
        ]

        self.reminder_tasks = [
            "call mom", "pay bills", "pick up dry cleaning", "buy groceries", "take medicine",
            "exercise", "water plants", "feed pets", "check email", "submit report",
            "schedule meeting", "book appointment", "return library books", "renew subscription"
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
            "Please book", "I need an appointment with", "Can I get an appointment with", "I'd like to book",
            "I want to book", "Can you schedule", "I need to book", "Please schedule me",
            "I'd like an appointment", "Can you make an appointment", "I need to schedule", "Book an appointment",
            "I want to see", "Can you set up", "I need a visit", "Please make an appointment",
            "I'd like to see", "Can you arrange", "I need to meet", "Book me an appointment",
            "I'm looking to book", "Can you help me book", "I need to schedule an appointment",
            "I want to make an appointment", "Can you set up an appointment", "I need to see a doctor",
            "I'd like to make an appointment", "Can you book me in", "I need to schedule a visit",
            "I want to book an appointment", "Can you arrange an appointment", "I need to see the doctor",
            "I'd like to schedule an appointment", "Can you make me an appointment", "I need to book a visit",
            "I want to schedule with", "Can you set up a meeting", "I need to make an appointment",
            "I'd like to book a visit", "Can you schedule me in", "I need to arrange an appointment"
        ]

        self.cancellation_phrases = [
            "Cancel my appointment", "I need to cancel", "Please cancel", "I can't make it",
            "I have to cancel", "Cancel the appointment", "I need to reschedule", "I can't attend",
            "Please reschedule", "I need to change", "I have to postpone", "Cancel my visit",
            "I want to cancel", "Can you cancel", "I need to cancel my visit", "Please cancel my appointment",
            "I have to cancel my appointment", "I can't make my appointment", "I need to cancel my meeting",
            "Please cancel my visit", "I want to cancel my appointment", "Can you cancel my visit",
            "I need to cancel my meeting", "I have to cancel my visit", "I can't attend my appointment"
        ]

        self.reschedule_phrases = [
            "I need to reschedule", "Can you reschedule", "I want to reschedule", "Please reschedule",
            "I need to change my appointment", "Can you change my appointment", "I want to change my appointment",
            "Please change my appointment", "I need to move my appointment", "Can you move my appointment",
            "I want to move my appointment", "Please move my appointment", "I need to postpone",
            "Can you postpone", "I want to postpone", "Please postpone", "I need to change the time",
            "Can you change the time", "I want to change the time", "Please change the time"
        ]

        self.find_doctor_phrases = [
            "I need to find", "Can you find", "I want to find", "Please find",
            "I'm looking for", "Can you help me find", "I need a", "I want a",
            "Please help me find", "I'm searching for", "Can you search for", "I need to see a",
            "I want to see a", "Please help me find a", "I'm looking to see", "Can you recommend a",
            "I need a recommendation for", "I want a recommendation for", "Please recommend a"
        ]

        self.medical_records_phrases = [
            "I need to see my medical records", "Can you show me my medical records", "I want to see my medical records",
            "Please show me my medical records", "I need my medical history", "Can you get my medical history",
            "I want my medical history", "Please get my medical history", "I need my health records",
            "Can you show me my health records", "I want my health records", "Please show me my health records",
            "I need to see my records", "Can you show me my records", "I want to see my records",
            "Please show me my records", "I need my medical information", "Can you get my medical information",
            "I want my medical information", "Please get my medical information",
            "I need to access my medical records", "Can you pull up my medical records", "I want to view my medical records",
            "Please access my medical records", "I need to check my medical history", "Can you retrieve my medical history",
            "I want to review my medical history", "Please retrieve my medical history", "I need to see my patient records",
            "Can you show me my patient records", "I want to see my patient records", "Please show me my patient records",
            "I need to access my health information", "Can you get my health information", "I want to see my health information",
            "Please get my health information", "I need to review my medical file", "Can you open my medical file",
            "I want to see my medical file", "Please open my medical file"
        ]

        self.prescription_phrases = [
            "I need a refill", "Can you refill", "I want a refill", "Please refill",
            "I need more", "Can you get me more", "I want more", "Please get me more",
            "I need to refill", "Can you help me refill", "I want to refill", "Please help me refill",
            "I need a prescription refill", "Can you refill my prescription", "I want a prescription refill",
            "Please refill my prescription", "I need more medication", "Can you get me more medication",
            "I want more medication", "Please get me more medication"
        ]

        self.update_phrases = [
            "I changed my", "My new", "Update my", "I need to update",
            "Please update", "Change my", "My contact info is now", "I moved to",
            "My new number is", "My new email is", "I got a new", "Please change my",
            "I need to change my", "Can you update my", "I want to update my", "Please change my",
            "I moved and need to update", "My phone number changed", "My email changed", "I got a new phone",
            "I need to update my contact information", "Can you change my contact information",
            "I want to update my contact information", "Please change my contact information"
        ]

    def create_system_message(self, available_tools: List[Dict] = None, conversation_type: str = None) -> Dict[str, str]:
        """Create the system message with tools"""
        if available_tools is None:
            available_tools = self.tools
            
        tools_json = json.dumps(available_tools, indent=2)
        
        # Create function list for display
        function_list = []
        for tool in available_tools:
            function_list.append(f"- {tool['name']}: {tool['description']}")
        
        function_list_text = "\n".join(function_list)
        
        # Different system prompt templates with sequential function calling instructions
        system_prompts = [
            # Medical-focused assistant
            f"""You are a medical AI assistant. Your job is to help users by understanding their request, identifying missing information if any, and then generating a structured plan using available tools.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # General AI assistant
            f"""You are an AI assistant designed to help users with various tasks. You can handle medical, personal, and professional requests. Your role is to understand user needs and use available tools to provide assistance.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Professional assistant
            f"""You are a professional AI assistant with expertise in healthcare and general productivity tasks. You help users by understanding their requests and using appropriate tools to complete tasks efficiently.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Conversational assistant
            f"""You are a helpful AI assistant that can handle a wide range of tasks. You're friendly, professional, and always try to understand what the user needs before taking action.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Task-oriented assistant
            f"""You are a task-oriented AI assistant. Your primary goal is to help users accomplish their goals by using the appropriate tools and functions available to you.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Healthcare-focused assistant
            f"""You are a healthcare AI assistant with capabilities for medical tasks and general assistance. You prioritize patient care and help users with both medical and non-medical requests.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Multi-purpose assistant
            f"""You are a multi-purpose AI assistant that can help with medical, personal, and professional tasks. You adapt your approach based on the user's needs and available tools.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Efficient assistant
            f"""You are an efficient AI assistant designed to quickly understand and execute user requests. You use available tools to get tasks done promptly and accurately.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Supportive assistant
            f"""You are a supportive AI assistant that helps users with their daily tasks and needs. You're patient, understanding, and always ready to assist with whatever they need.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}""",

            # Expert assistant
            f"""You are an expert AI assistant with specialized knowledge in healthcare and general assistance. You provide accurate, helpful responses and use tools effectively to meet user needs.

# Important Instructions
- Output only ONE function call at a time
- Wait for the function response before making additional calls
- If multiple functions are needed, make them sequentially
- Always provide clear, helpful responses after receiving function results

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
{function_list_text}"""
        ]
        
        # Select system prompt based on conversation type or randomly
        if conversation_type == "medical":
            # Prefer medical-focused prompts for medical conversations
            medical_prompts = [system_prompts[0], system_prompts[5], system_prompts[8]]  # Medical, Healthcare, Expert
            system_content = random.choice(medical_prompts)
        elif conversation_type == "general":
            # Prefer general prompts for non-medical conversations
            general_prompts = [system_prompts[1], system_prompts[3], system_prompts[6]]  # General, Conversational, Multi-purpose
            system_content = random.choice(general_prompts)
        else:
            # Random selection for mixed or unknown types
            system_content = random.choice(system_prompts)

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

    def generate_simple_booking_example(self) -> Dict[str, Any]:
        """Generate example: Simple booking with System -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to gather the user's email for appointment confirmation.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_booking_example(self) -> Dict[str, Any]:
        """Generate example: Extended booking with System -> User -> Assistant -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to gather the user's email for appointment confirmation.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My email is john.doe@email.com"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the email. I can proceed with booking the appointment.\n</think>\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"{time}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_simple_cancellation_example(self) -> Dict[str, Any]:
        """Generate example: Simple cancellation with System -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        appointment_id = random.choice(self.appointment_ids)
        phrase = random.choice(self.cancellation_phrases)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = random.choice(days)
        user_input = f"{phrase} with {doctor} scheduled for {day}."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the appointment ID to cancel the appointment.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_cancellation_example(self) -> Dict[str, Any]:
        """Generate example: Extended cancellation with System -> User -> Assistant -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        appointment_id = random.choice(self.appointment_ids)
        phrase = random.choice(self.cancellation_phrases)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = random.choice(days)
        user_input = f"{phrase} with {doctor} scheduled for {day}."

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the appointment ID to cancel the appointment.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My appointment ID is {appointment_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the appointment ID. I can proceed with cancelling the appointment.\n</think>\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"cancelled\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_direct_booking_example(self) -> Dict[str, Any]:
        """Generate example: Direct booking with System -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI have all the information needed. I can book the appointment directly.\n</think>\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"{time}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_reschedule_example(self) -> Dict[str, Any]:
        """Generate example: Reschedule appointment with System -> User -> Assistant"""
        appointment_id = random.choice(self.appointment_ids)
        new_time = random.choice(self.time_formats)
        user_input = f"I need to reschedule my appointment to {new_time}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the appointment ID to reschedule the appointment.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_reschedule_example(self) -> Dict[str, Any]:
        """Generate example: Extended reschedule with System -> User -> Assistant -> User -> Assistant"""
        appointment_id = random.choice(self.appointment_ids)
        new_time = random.choice(self.time_formats)
        user_input = f"I need to reschedule my appointment to {new_time}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the appointment ID to reschedule the appointment.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My appointment ID is {appointment_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the appointment ID. I can proceed with rescheduling.\n</think>\n<tool_call>\n{{\"name\": \"reschedule_appointment\", \"arguments\": {{\"appointment_id\": \"{appointment_id}\", \"new_time\": \"{new_time}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_find_doctor_example(self) -> Dict[str, Any]:
        """Generate example: Find doctor by specialty with System -> User -> Assistant"""
        specialty = random.choice(self.specialties)
        user_input = f"I need to find a {specialty}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to find a doctor by specialty.\n</think>\n<tool_call>\n{{\"name\": \"find_doctor\", \"arguments\": {{\"specialty\": \"{specialty}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_medical_records_example(self) -> Dict[str, Any]:
        """Generate example: Get medical records with System -> User -> Assistant"""
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need to see my medical records"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve medical records.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_medical_records_example(self) -> Dict[str, Any]:
        """Generate example: Extended medical records with System -> User -> Assistant -> User -> Assistant"""
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need to see my medical records"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve medical records.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can retrieve the medical records.\n</think>\n<tool_call>\n{{\"name\": \"get_medical_records\", \"arguments\": {{\"patient_id\": \"{patient_id}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_prescription_refill_example(self) -> Dict[str, Any]:
        """Generate example: Prescription refill with System -> User -> Assistant"""
        medication = random.choice(self.medications)
        user_input = f"I need a refill for {medication}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to request a prescription refill.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_prescription_refill_example(self) -> Dict[str, Any]:
        """Generate example: Extended prescription refill with System -> User -> Assistant -> User -> Assistant"""
        medication = random.choice(self.medications)
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need a refill for {medication}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to request a prescription refill.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can request the prescription refill.\n</think>\n<tool_call>\n{{\"name\": \"request_prescription_refill\", \"arguments\": {{\"patient_id\": \"{patient_id}\", \"medication\": \"{medication}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_urgent_booking_example(self) -> Dict[str, Any]:
        """Generate example: Urgent booking with System -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        user_input = f"I need to see {doctor} ASAP"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user needs an urgent appointment. I should check for the earliest available time.\n</think>\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"ASAP\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_specialty_search_example(self) -> Dict[str, Any]:
        """Generate example: Find doctor by specialty with System -> User -> Assistant"""
        specialty = random.choice(self.specialties)
        phrase = random.choice(self.find_doctor_phrases)
        user_input = f"{phrase} {specialty}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to find a doctor by specialty.\n</think>\n<tool_call>\n{{\"name\": \"find_doctor\", \"arguments\": {{\"specialty\": \"{specialty}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_availability_check_example(self) -> Dict[str, Any]:
        """Generate example: Check doctor availability with System -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        date_range = random.choice(self.date_ranges)
        user_input = f"Is {doctor} available {date_range}?"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to check the doctor's availability for the specified time period.\n</think>\n<tool_call>\n{{\"name\": \"find_doctor\", \"arguments\": {{\"specialty\": \"{doctor}\", \"availability\": \"{date_range}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_contact_update_example(self) -> Dict[str, Any]:
        """Generate example: Update contact information with System -> User -> Assistant"""
        fields = ["phone", "email", "address"]
        field = random.choice(fields)
        phrase = random.choice(self.update_phrases)
        
        if field == "phone":
            value = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif field == "email":
            domains = ["gmail.com", "yahoo.com", "outlook.com", "healthcare.com"]
            value = f"patient{random.randint(1, 999)}@{random.choice(domains)}"
        else:
            streets = ["Main St", "Oak Ave", "Pine Rd", "Elm St", "Cedar Ln"]
            value = f"{random.randint(100, 9999)} {random.choice(streets)}"

        user_input = f"{phrase} {field} to {value}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to update their contact information.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_contact_update_example(self) -> Dict[str, Any]:
        """Generate example: Extended contact update with System -> User -> Assistant -> User -> Assistant"""
        fields = ["phone", "email", "address"]
        field = random.choice(fields)
        patient_id = random.choice(self.patient_ids)
        phrase = random.choice(self.update_phrases)
        
        if field == "phone":
            value = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif field == "email":
            domains = ["gmail.com", "yahoo.com", "outlook.com", "healthcare.com"]
            value = f"patient{random.randint(1, 999)}@{random.choice(domains)}"
        else:
            streets = ["Main St", "Oak Ave", "Pine Rd", "Elm St", "Cedar Ln"]
            value = f"{random.randint(100, 9999)} {random.choice(streets)}"

        user_input = f"{phrase} {field} to {value}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to update their contact information.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can update their contact information.\n</think>\n<tool_call>\n{{\"name\": \"update_patient_info\", \"arguments\": {{\"patient_id\": \"{patient_id}\", \"field\": \"{field}\", \"new_value\": \"{value}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_emergency_cancellation_example(self) -> Dict[str, Any]:
        """Generate example: Emergency cancellation with System -> User -> Assistant"""
        user_input = "I have an emergency and need to cancel my appointment"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user has an emergency and needs to cancel. I should get the appointment ID to cancel it.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_follow_up_booking_example(self) -> Dict[str, Any]:
        """Generate example: Follow-up appointment booking with System -> User -> Assistant"""
        doctor = random.choice(self.doctors)
        user_input = f"I need a follow-up appointment with {doctor}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user needs a follow-up appointment. I should get their email for confirmation.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_test_results_example(self) -> Dict[str, Any]:
        """Generate example: Get test results with System -> User -> Assistant"""
        test_type = random.choice(self.test_types)
        user_input = f"I need to see my {test_type} results"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve test results.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_test_results_example(self) -> Dict[str, Any]:
        """Generate example: Extended test results with System -> User -> Assistant -> User -> Assistant"""
        test_type = random.choice(self.test_types)
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need to see my {test_type} results"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve test results.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can retrieve the test results.\n</think>\n<tool_call>\n{{\"name\": \"get_test_results\", \"arguments\": {{\"patient_id\": \"{patient_id}\", \"test_type\": \"{test_type}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_schedule_lab_test_example(self) -> Dict[str, Any]:
        """Generate example: Schedule lab test with System -> User -> Assistant"""
        test_type = random.choice(self.test_types)
        user_input = f"I need to schedule a {test_type}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to schedule the lab test.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_schedule_lab_test_example(self) -> Dict[str, Any]:
        """Generate example: Extended schedule lab test with System -> User -> Assistant -> User -> Assistant"""
        test_type = random.choice(self.test_types)
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need to schedule a {test_type}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to schedule the lab test.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can schedule the lab test.\n</think>\n<tool_call>\n{{\"name\": \"schedule_lab_test\", \"arguments\": {{\"patient_id\": \"{patient_id}\", \"test_type\": \"{test_type}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_get_medication_list_example(self) -> Dict[str, Any]:
        """Generate example: Get medication list with System -> User -> Assistant"""
        user_input = "I need to see my current medications"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve their medication list.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_get_medication_list_example(self) -> Dict[str, Any]:
        """Generate example: Extended get medication list with System -> User -> Assistant -> User -> Assistant"""
        patient_id = random.choice(self.patient_ids)
        user_input = "I need to see my current medications"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve their medication list.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can retrieve their medication list.\n</think>\n<tool_call>\n{{\"name\": \"get_medication_list\", \"arguments\": {{\"patient_id\": \"{patient_id}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_add_medication_example(self) -> Dict[str, Any]:
        """Generate example: Add medication with System -> User -> Assistant"""
        medication = random.choice(self.medications)
        dosage = random.choice(self.dosages)
        frequency = random.choice(self.medication_frequencies)
        user_input = f"I need to add {medication} to my medication list"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to add medication to their list.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_add_medication_example(self) -> Dict[str, Any]:
        """Generate example: Extended add medication with System -> User -> Assistant -> User -> Assistant"""
        medication = random.choice(self.medications)
        dosage = random.choice(self.dosages)
        frequency = random.choice(self.medication_frequencies)
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need to add {medication} to my medication list"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to add medication to their list.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can add the medication to their list.\n</think>\n<tool_call>\n{{\"name\": \"add_medication\", \"arguments\": {{\"patient_id\": \"{patient_id}\", \"medication_name\": \"{medication}\", \"dosage\": \"{dosage}\", \"frequency\": \"{frequency}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_get_allergies_example(self) -> Dict[str, Any]:
        """Generate example: Get allergies with System -> User -> Assistant"""
        user_input = "I need to see my allergy information"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve their allergy information.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_add_allergy_example(self) -> Dict[str, Any]:
        """Generate example: Add allergy with System -> User -> Assistant"""
        allergy = random.choice(self.allergies)
        severity = random.choice(self.allergy_severities)
        user_input = f"I need to add {allergy} to my allergy list"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to add allergy to their record.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_add_allergy_example(self) -> Dict[str, Any]:
        """Generate example: Extended add allergy with System -> User -> Assistant -> User -> Assistant"""
        allergy = random.choice(self.allergies)
        severity = random.choice(self.allergy_severities)
        patient_id = random.choice(self.patient_ids)
        user_input = f"I need to add {allergy} to my allergy list"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to add allergy to their record.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"My patient ID is {patient_id}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the patient ID. I can add the allergy to their record.\n</think>\n<tool_call>\n{{\"name\": \"add_allergy\", \"arguments\": {{\"patient_id\": \"{patient_id}\", \"allergy_name\": \"{allergy}\", \"severity\": \"{severity}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_get_insurance_example(self) -> Dict[str, Any]:
        """Generate example: Get insurance info with System -> User -> Assistant"""
        user_input = "I need to see my insurance information"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to retrieve their insurance information.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_update_insurance_example(self) -> Dict[str, Any]:
        """Generate example: Update insurance with System -> User -> Assistant"""
        provider = random.choice(self.insurance_providers)
        user_input = f"I need to update my insurance to {provider}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI need to get the patient ID to update their insurance information.\n</think>\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_booking_example(self) -> Dict[str, Any]:
        """Generate example: Booking request when booking is not available"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        # Create limited tools without booking capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['book_appointment', 'cancel_appointment', 'reschedule_appointment']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting to book an appointment, but I don't have access to appointment booking functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to book appointments. However, I can help you with other medical administrative tasks such as:\n\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy information\n- Handling insurance information\n\nWould you like help with any of these other services?"
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_test_results_example(self) -> Dict[str, Any]:
        """Generate example: Test results request when not available"""
        test_type = random.choice(self.test_types)
        user_input = f"I need to see my {test_type} results"

        # Create limited tools without test results capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['get_test_results', 'schedule_lab_test']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting test results, but I don't have access to test results functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have access to test results or lab scheduling functionality. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Managing allergy information\n- Handling insurance information\n\nWould you like help with any of these other services?"
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_medication_example(self) -> Dict[str, Any]:
        """Generate example: Medication request when not available"""
        medication = random.choice(self.medications)
        user_input = f"I need a refill for {medication}"

        # Create limited tools without medication capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['request_prescription_refill', 'get_medication_list', 'add_medication']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting a prescription refill, but I don't have access to medication management functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have access to medication management or prescription refill functionality. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Updating contact information\n- Getting test results\n- Managing allergy information\n- Handling insurance information\n\nWould you like help with any of these other services?"
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_insurance_example(self) -> Dict[str, Any]:
        """Generate example: Insurance request when not available"""
        provider = random.choice(self.insurance_providers)
        user_input = f"I need to update my insurance to {provider}"

        # Create limited tools without insurance capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['get_insurance_info', 'update_insurance_info']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting to update insurance information, but I don't have access to insurance management functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have access to insurance management functionality. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy information\n\nWould you like help with any of these other services?"
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_allergy_example(self) -> Dict[str, Any]:
        """Generate example: Allergy request when not available"""
        allergy = random.choice(self.allergies)
        user_input = f"I need to add {allergy} to my allergy list"

        # Create limited tools without allergy capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['get_allergies', 'add_allergy']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting to add allergy information, but I don't have access to allergy management functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have access to allergy management functionality. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Handling insurance information\n\nWould you like help with any of these other services?"
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_medical_records_example(self) -> Dict[str, Any]:
        """Generate example: Medical records request when not available"""
        user_input = "I need to see my medical records"

        # Create limited tools without medical records capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['get_medical_records']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting medical records, but I don't have access to medical records functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have access to medical records functionality. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy information\n- Handling insurance information\n\nWould you like help with any of these other services?"
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_multi_turn_complex_example(self) -> Dict[str, Any]:
        """Generate example: Multi-turn complex conversation with multiple functions"""
        doctor = random.choice(self.doctors)
        medication = random.choice(self.medications)
        test_type = random.choice(self.test_types)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I need to book an appointment with {doctor} and also get my {test_type} results"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user has two requests: booking an appointment and getting test results. I need to handle both requests systematically.\n</think>\nI'll help you with both requests. Let me start by getting your email for the appointment booking.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My email is john.doe@email.com"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the email. I can book the appointment and then get the test results.\n</think>\nGreat! I'll book your appointment with {doctor}.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"next available\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_id\": \"apt_12345\", \"status\": \"confirmed\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe appointment is booked. Now I need to get the test results.\n</think>\nPerfect! Your appointment is confirmed. Now for your {test_type} results, I need your patient ID.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_context_switching_example(self) -> Dict[str, Any]:
        """Generate example: Context switching between different medical tasks"""
        doctor = random.choice(self.doctors)
        allergy = random.choice(self.allergies)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I need to cancel my appointment with {doctor} and also add {allergy} to my allergy list"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to cancel an appointment and add an allergy. I need to handle both requests.\n</think>\nI'll help you with both requests. First, let me get your appointment ID to cancel the appointment.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"appointment_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My appointment ID is apt_12345"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I can cancel the appointment and then handle the allergy request.\n</think>\nI'll cancel your appointment now.\n\n<tool_call>\n{{\"name\": \"cancel_appointment\", \"arguments\": {{\"appointment_id\": \"apt_12345\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"cancelled\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe appointment is cancelled. Now I need to add the allergy to their record.\n</think>\nYour appointment has been cancelled. Now for adding {allergy} to your allergy list, I need your patient ID.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_error_recovery_example(self) -> Dict[str, Any]:
        """Generate example: Error recovery and clarification"""
        doctor = random.choice(self.doctors)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I want to see {doctor}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to see a doctor but hasn't specified what they want to do (book appointment, check availability, etc.). I should ask for clarification.\n</think>\nI'd be happy to help you with {doctor}. Could you please clarify what you'd like to do? For example:\n\n- Book an appointment\n- Check availability\n- Get information about the doctor\n- Cancel an existing appointment\n\nWhat would you like to do?"
                },
                {
                    "role": "user",
                    "content": "I want to book an appointment"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow the user wants to book an appointment. I need to get their email for confirmation.\n</think>\nPerfect! I'll help you book an appointment with {doctor}. I need your email address to send you a confirmation.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_priority_handling_example(self) -> Dict[str, Any]:
        """Generate example: Priority handling for urgent vs routine requests"""
        doctor = random.choice(self.doctors)
        medication = random.choice(self.medications)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I need a refill for {medication} and also want to book a routine checkup with {doctor}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user has two requests: a medication refill (urgent) and a routine checkup (less urgent). I should prioritize the medication refill first.\n</think>\nI'll help you with both requests. Since medication refills are time-sensitive, let me handle that first. I need your patient ID to request the refill.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My patient ID is user_12345"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I can request the medication refill and then handle the appointment booking.\n</think>\nI'll request your {medication} refill now.\n\n<tool_call>\n{{\"name\": \"request_prescription_refill\", \"arguments\": {{\"patient_id\": \"user_12345\", \"medication\": \"{medication}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"refill_requested\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe medication refill is requested. Now I can handle the routine checkup booking.\n</think>\nGreat! Your {medication} refill has been requested. Now for the routine checkup with {doctor}, I need your email for appointment confirmation.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_email\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_web_search_example(self) -> Dict[str, Any]:
        """Generate example: Web search request"""
        search_topics = [
            "latest news about AI", "weather forecast", "restaurant reviews", "movie showtimes",
            "flight prices", "hotel availability", "shopping deals", "sports scores",
            "stock market news", "cooking recipes", "travel guides", "product reviews"
        ]
        topic = random.choice(search_topics)
        user_input = f"Can you search for {topic}?"

        # Different response styles
        response_styles = [
            f"<think>\nThe user wants me to search the web for information about {topic}.\n</think>\nI'll search the web for information about {topic}.\n\n<tool_call>\n{{\"name\": \"search_web\", \"arguments\": {{\"query\": \"{topic}\", \"search_type\": \"general\"}}}}\n</tool_call>",
            
            f"<think>\nThe user is asking me to search for {topic}. I should use the web search function.\n</think>\nSure! Let me search for {topic} for you.\n\n<tool_call>\n{{\"name\": \"search_web\", \"arguments\": {{\"query\": \"{topic}\", \"search_type\": \"general\"}}}}\n</tool_call>",
            
            f"<think>\nI need to search the web for {topic} as requested by the user.\n</think>\nI'll help you find information about {topic}.\n\n<tool_call>\n{{\"name\": \"search_web\", \"arguments\": {{\"query\": \"{topic}\", \"search_type\": \"general\"}}}}\n</tool_call>",
            
            f"<think>\nThe user wants information about {topic}. I should search the web.\n</think>\nLet me search for that information for you.\n\n<tool_call>\n{{\"name\": \"search_web\", \"arguments\": {{\"query\": \"{topic}\", \"search_type\": \"general\"}}}}\n</tool_call>"
        ]

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": random.choice(response_styles)
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_email_example(self) -> Dict[str, Any]:
        """Generate example: Send email request"""
        user_input = "I need to send an email to my boss about the project update"

        # Different tool call formats
        tool_call_formats = [
            f"<think>\nThe user wants to send an email to their boss about a project update. I need to get the email details.\n</think>\nI'll help you send that email. I need the recipient's email address.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"to\"}}}}\n</tool_call>",
            
            f"<think>\nThe user needs to send an email. I should ask for the recipient's email address.\n</think>\nI can help you send that email. What's your boss's email address?\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"to\"}}}}\n</tool_call>",
            
            f"<think>\nI need to get the email address to send the email.\n</think>\nTo send the email, I'll need the recipient's email address.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"to\"}}}}\n</tool_call>",
            
            f"<think>\nThe user wants to send an email. I need the recipient email.\n</think>\nI'll help you send the email. Could you provide the recipient's email address?\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"to\"}}}}\n</tool_call>"
        ]

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": random.choice(tool_call_formats)
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_extended_email_example(self) -> Dict[str, Any]:
        """Generate example: Extended email with System -> User -> Assistant -> User -> Assistant"""
        user_input = "I need to send an email to my boss about the project update"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to send an email to their boss about a project update. I need to get the email details.\n</think>\nI'll help you send that email. I need the recipient's email address.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"to\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My boss's email is boss@company.com"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nNow I have the email address. I can send the email.\n</think>\nPerfect! I'll send the email now.\n\n<tool_call>\n{{\"name\": \"send_email\", \"arguments\": {{\"to\": \"boss@company.com\", \"subject\": \"Project Update\", \"body\": \"Hi, here's the latest project update as requested.\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_calendar_event_example(self) -> Dict[str, Any]:
        """Generate example: Create calendar event"""
        event_type = random.choice(self.event_types)
        user_input = f"I need to schedule a {event_type} for tomorrow"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to schedule a {event_type} for tomorrow. I need to get the event details.\n</think>\nI'll help you schedule that {event_type}. What time would you like it for?\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"time\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_reminder_example(self) -> Dict[str, Any]:
        """Generate example: Set reminder"""
        task = random.choice(self.reminder_tasks)
        user_input = f"I need a reminder to {task}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to set a reminder for {task}. I need to get the date and time.\n</think>\nI'll set a reminder for you to {task}. When would you like to be reminded?\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"date\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_translation_example(self) -> Dict[str, Any]:
        """Generate example: Translate text"""
        language = random.choice(self.languages)
        user_input = f"Can you translate 'Hello, how are you?' to {language}?"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants me to translate text to {language}.\n</think>\nI'll translate that text to {language} for you.\n\n<tool_call>\n{{\"name\": \"translate_text\", \"arguments\": {{\"text\": \"Hello, how are you?\", \"target_language\": \"{language}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_math_calculation_example(self) -> Dict[str, Any]:
        """Generate example: Math calculation"""
        math_expressions = [
            "15 + 27", "100 - 45", "12 * 8", "144 / 12", "5^2", "sqrt(81)",
            "25% of 200", "15 + 27 * 3", "(10 + 5) * 2", "1000 / 4 + 50"
        ]
        expression = random.choice(math_expressions)
        user_input = f"Can you calculate {expression}?"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants me to calculate the mathematical expression {expression}.\n</think>\nI'll calculate {expression} for you.\n\n<tool_call>\n{{\"name\": \"calculate_math\", \"arguments\": {{\"expression\": \"{expression}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_weather_example(self) -> Dict[str, Any]:
        """Generate example: Get weather information"""
        city = random.choice(self.cities)
        user_input = f"What's the weather like in {city}?"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to know the weather in {city}.\n</think>\nI'll get the weather information for {city}.\n\n<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"location\": \"{city}\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_travel_booking_example(self) -> Dict[str, Any]:
        """Generate example: Book travel"""
        destination = random.choice(self.travel_destinations)
        travel_type = random.choice(self.travel_types)
        user_input = f"I want to book a {travel_type} to {destination}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to book a {travel_type} to {destination}. I need to get the travel date.\n</think>\nI'll help you book a {travel_type} to {destination}. When would you like to travel?\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"date\"}}}}\n</tool_call>"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_food_order_example(self) -> Dict[str, Any]:
        """Generate example: Order food"""
        restaurant = random.choice(self.restaurants)
        food_item = random.choice(self.food_items)
        user_input = f"I want to order {food_item} from {restaurant}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user wants to order {food_item} from {restaurant}. I can place the order.\n</think>\nI'll help you order {food_item} from {restaurant}.\n\n<tool_call>\n{{\"name\": \"order_food\", \"arguments\": {{\"restaurant\": \"{restaurant}\", \"items\": \"{food_item}\"}}}}\n</tool_call>"
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

    def get_random_tools(self, min_tools: int = 3, max_tools: int = 15) -> List[Dict]:
        """Get a random subset of tools for each conversation"""
        # Ensure min_tools is not greater than max_tools
        if min_tools > max_tools:
            min_tools, max_tools = max_tools, min_tools
        
        # Ensure we don't try to sample more tools than available
        max_tools = min(max_tools, len(self.tools))
        min_tools = min(min_tools, max_tools)
        
        # If min_tools equals max_tools, use that value directly
        if min_tools == max_tools:
            num_tools = min_tools
        else:
            num_tools = random.randint(min_tools, max_tools)
        
        return random.sample(self.tools, num_tools)

    def generate_dataset(self, num_examples: int = 800) -> List[Dict[str, Any]]:
        """Generate a complete dataset with System->User->Assistant conversations for function calling"""
        dataset = []

        # Define example generators with increased variance
        generators = [
            # Simple booking variations (18%)
            (self.generate_simple_booking_example, 0.18),
            
            # Simple cancellation variations (12%)
            (self.generate_simple_cancellation_example, 0.12),
            
            # Doctor search variations (10%)
            (self.generate_find_doctor_example, 0.10),
            
            # Medical records variations (8%)
            (self.generate_medical_records_example, 0.08),
            
            # Prescription refill variations (8%)
            (self.generate_prescription_refill_example, 0.08),
            
            # Web search variations (8%)
            (self.generate_web_search_example, 0.04),
            (self.generate_email_example, 0.04),
            
            # Direct booking (4%)
            (self.generate_direct_booking_example, 0.04),
            
            # Rescheduling variations (6%)
            (self.generate_reschedule_example, 0.03),
            (self.generate_extended_reschedule_example, 0.03),
            
            # Test results variations (4%)
            (self.generate_test_results_example, 0.02),
            (self.generate_extended_test_results_example, 0.02),
            
            # Lab test scheduling (4%)
            (self.generate_schedule_lab_test_example, 0.02),
            (self.generate_extended_schedule_lab_test_example, 0.02),
            
            # Medication management (4%)
            (self.generate_get_medication_list_example, 0.02),
            (self.generate_add_medication_example, 0.02),
            
            # Contact updates (3%)
            (self.generate_contact_update_example, 0.02),
            (self.generate_extended_contact_update_example, 0.01),
            
            # Allergy management (3%)
            (self.generate_get_allergies_example, 0.02),
            (self.generate_add_allergy_example, 0.01),
            
            # Insurance management (3%)
            (self.generate_get_insurance_example, 0.02),
            (self.generate_update_insurance_example, 0.01),
            
            # Urgent/emergency requests (3%)
            (self.generate_urgent_booking_example, 0.02),
            (self.generate_emergency_cancellation_example, 0.01),
            
            # General assistant tasks (4%)
            (self.generate_calendar_event_example, 0.01),
            (self.generate_reminder_example, 0.01),
            (self.generate_translation_example, 0.01),
            (self.generate_math_calculation_example, 0.01),
            
            # Unsupported requests (12%)
            (self.generate_unsupported_booking_example, 0.02),
            (self.generate_unsupported_test_results_example, 0.01),
            (self.generate_unsupported_medication_example, 0.01),
            (self.generate_unsupported_web_search_example, 0.01),
            (self.generate_unsupported_email_example, 0.01),
            (self.generate_unsupported_calendar_example, 0.01),
            (self.generate_unsupported_translation_example, 0.01),
            (self.generate_unsupported_weather_example, 0.01),
            (self.generate_unsupported_travel_booking_example, 0.01),
            (self.generate_unsupported_food_order_example, 0.01),
            (self.generate_unsupported_medical_records_example, 0.01),
            
            # No-tools interactions (6%)
            (self.generate_no_tools_general_help_example, 0.02),
            (self.generate_no_tools_appointment_reminder_example, 0.02),
            (self.generate_no_tools_medication_reminder_example, 0.01),
            (self.generate_no_tools_health_tips_example, 0.01),
            
            # One-shot examples with complete tool calls (20%)
            (self.generate_one_shot_booking_example, 0.03),
            (self.generate_one_shot_find_doctor_example, 0.03),
            (self.generate_one_shot_web_search_example, 0.02),
            (self.generate_one_shot_medical_records_example, 0.02),
            (self.generate_one_shot_prescription_refill_example, 0.02),
            (self.generate_one_shot_complex_workflow_example, 0.02),
            (self.generate_one_shot_unsupported_request_example, 0.01),
            (self.generate_one_shot_error_recovery_example, 0.01),
            (self.generate_one_shot_medication_management_example, 0.02),
            (self.generate_one_shot_contact_insurance_update_example, 0.02)
        ]

        # Generate examples with shuffling for better randomization
        for _ in range(num_examples):
            # Select generator based on weights
            generator, weight = random.choices(generators, weights=[w for _, w in generators])[0]
            example = generator()
            
            # For one-shot examples, keep the full conversation (5+ messages)
            # For simple examples, ensure System->User->Assistant format (3 messages)
            if "one_shot" not in generator.__name__ and len(example['messages']) > 3:
                # Keep only the first 3 messages (System, User, Assistant)
                example['messages'] = example['messages'][:3]
            
            # Add variance by randomly selecting tools with more variety
            # Skip tool randomization for no-tools examples and unsupported examples
            if (not example['messages'][0]['content'].startswith('I apologize, but I don\'t have') and 
                not example['messages'][0]['content'].startswith('You are a helpful medical assistant') and
                len(example['tools']) > 0):
                # Get random tools for this conversation with more variance
                min_tools = random.choice([2, 3, 4, 5, 6])
                max_tools = random.choice([5, 6, 7, 8, 9, 10, 12, 15])
                random_tools = self.get_random_tools(min_tools, max_tools)
                # Update the system message with random tools
                example['messages'][0] = self.create_system_message(random_tools)
                example['tools'] = random_tools
            
            dataset.append(example)

        # Multiple rounds of shuffling for better randomization
        print(f"Generated {len(dataset)} examples. Shuffling dataset...")
        
        # First shuffle: Basic randomization
        random.shuffle(dataset)
        
        # Second shuffle: Additional randomization for better distribution
        random.shuffle(dataset)
        
        # Third shuffle: Final randomization to ensure no patterns
        random.shuffle(dataset)
        
        print("Dataset shuffling completed.")
        
        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "nad_training_dataset.json"):
        """Save dataset to JSON file with additional shuffling"""
        print(f"Final shuffling before saving {len(dataset)} examples...")
        
        # Additional shuffling before saving to ensure maximum randomization
        random.shuffle(dataset)
        random.shuffle(dataset)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename} with {len(dataset)} examples")

    def generate_test_dataset(self, num_examples: int = 200) -> List[Dict[str, Any]]:
        """Generate a test dataset with enhanced shuffling for evaluation"""
        print(f"Generating test dataset with {num_examples} examples...")
        
        # Generate the dataset using the same logic as training dataset
        dataset = self.generate_dataset(num_examples)
        
        # Additional shuffling specifically for test dataset
        print("Applying enhanced shuffling for test dataset...")
        
        # Multiple rounds of shuffling with different seeds
        random.seed(42)  # Set seed for reproducible shuffling
        random.shuffle(dataset)
        
        random.seed(123)  # Different seed
        random.shuffle(dataset)
        
        random.seed()  # Reset to random seed
        random.shuffle(dataset)
        random.shuffle(dataset)
        
        print(f"Test dataset generated with {len(dataset)} examples and enhanced shuffling.")
        return dataset

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
        unsupported_requests = 0
        supported_requests = 0

        for example in dataset:
            # Check if this is an unsupported capability example
            is_unsupported = any('I apologize, but I don\'t have' in msg['content'] 
                               for msg in example['messages'] if msg['role'] == 'assistant')
            
            if is_unsupported:
                unsupported_requests += 1
            else:
                supported_requests += 1

            for message in example['messages']:
                if message['role'] == 'assistant' and '<tool_call>' in message['content']:
                    # Count tool calls in the content
                    content = message['content']
                    for tool in self.tools:
                        if f'"name": "{tool["name"]}"' in content:
                            tool_counts[tool['name']] = tool_counts.get(tool['name'], 0) + 1
                            total_tool_calls += 1

        print(f"Total examples: {len(dataset)}")
        print(f"Supported requests: {supported_requests}")
        print(f"Unsupported requests: {unsupported_requests}")
        print(f"Total tool calls: {total_tool_calls}")
        print(f"Unique tools used: {len(tool_counts)}")
        
        print("\nTool usage:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} times")

        # Average tool calls per example
        avg_tool_calls = total_tool_calls / len(dataset) if dataset else 0
        print(f"\nAverage tool calls per example: {avg_tool_calls:.2f}")

        # Tool availability statistics
        tool_availability = {}
        for example in dataset:
            num_tools = len(example['tools'])
            tool_availability[num_tools] = tool_availability.get(num_tools, 0) + 1
        
        print(f"\nTool availability distribution:")
        for num_tools, count in sorted(tool_availability.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {num_tools} tools: {count} examples ({percentage:.1f}%)")

        # User input length statistics
        input_lengths = []
        for example in dataset:
            for message in example['messages']:
                if message['role'] == 'user':
                    input_lengths.append(len(message['content']))

        if input_lengths:
            avg_input_length = sum(input_lengths) / len(input_lengths)
            print(f"\nAverage user input length: {avg_input_length:.1f} characters")

    def generate_one_shot_booking_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complete booking with tool call and response"""
        doctor = random.choice(self.doctors)
        time = random.choice(self.time_formats)
        phrase = random.choice(self.booking_phrases)
        user_input = f"{phrase} {doctor} {time}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you book an appointment with {doctor} at {time}.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"{time}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_id\": \"apt_12345\", \"status\": \"confirmed\", \"confirmation_sent\": true, \"message\": \"Appointment successfully booked!\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": "Perfect! Your appointment has been successfully booked. Here's a summary:\n\n✅ **Appointment Confirmed**: {doctor} at {time}\n✅ **Appointment ID**: apt_12345\n✅ **Confirmation**: Email sent\n\nYou'll receive a confirmation email shortly with all the details. Please arrive 10 minutes before your scheduled time."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_find_doctor_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complete doctor search with tool call and response"""
        specialty = random.choice(self.specialties)
        phrase = random.choice(self.find_doctor_phrases)
        user_input = f"{phrase} {specialty}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you find a {specialty}.\n\n<tool_call>\n{{\"name\": \"find_doctor\", \"arguments\": {{\"specialty\": \"{specialty}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"doctors\": [{\"name\": \"Dr. Sarah Johnson\", \"specialty\": \"{specialty}\", \"availability\": \"Next week\", \"rating\": 4.8}, {\"name\": \"Dr. Michael Chen\", \"specialty\": \"{specialty}\", \"availability\": \"This week\", \"rating\": 4.9}], \"total_found\": 2}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": "I found 2 {specialty}s available:\n\n1. **Dr. Sarah Johnson**\n   - Availability: Next week\n   - Rating: 4.8/5\n\n2. **Dr. Michael Chen**\n   - Availability: This week\n   - Rating: 4.9/5\n\nWould you like me to book an appointment with either of these doctors?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_web_search_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complete web search with tool call and response"""
        search_queries = [
            "symptoms of diabetes", "healthy diet tips", "exercise benefits", "stress management",
            "sleep hygiene", "heart disease prevention", "mental health resources"
        ]
        query = random.choice(search_queries)
        user_input = f"Search for {query}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"I'll search for information about {query}.\n\n<tool_call>\n{{\"name\": \"search_web\", \"arguments\": {{\"query\": \"{query}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"results\": [{\"title\": \"{query.title()} - Comprehensive Guide\", \"snippet\": \"Learn about the key aspects of {query} including symptoms, prevention, and treatment options.\", \"url\": \"https://example.com/{query.replace(' ', '-')}\"}, {\"title\": \"Medical Information: {query.title()}\", \"snippet\": \"Expert medical advice and latest research on {query}.\", \"url\": \"https://medical.example.com/{query.replace(' ', '-')}\"}], \"total_results\": 2}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": "Here's what I found about {query}:\n\n1. **{query.title()} - Comprehensive Guide**\n   - Learn about the key aspects of {query} including symptoms, prevention, and treatment options.\n   - [Read more](https://example.com/{query.replace(' ', '-')})\n\n2. **Medical Information: {query.title()}**\n   - Expert medical advice and latest research on {query}.\n   - [Read more](https://medical.example.com/{query.replace(' ', '-')})\n\nThese resources provide comprehensive information about {query}. Would you like me to search for more specific aspects of this topic?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_medical_records_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complete medical records retrieval with sequential tool calls"""
        user_input = "I need my medical records"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": "I'll help you retrieve your medical records. I need your patient ID to access your information.\n\n<tool_call>\n{\"name\": \"request_missing_argument\", \"arguments\": {\"missing_field\": \"patient_id\"}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My patient ID is user_00123"
                },
                {
                    "role": "assistant",
                    "content": "Thank you! Now I'll retrieve your medical records.\n\n<tool_call>\n{\"name\": \"get_medical_records\", \"arguments\": {\"patient_id\": \"user_00123\"}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"patient_id\": \"user_00123\", \"records\": {\"last_visit\": \"2024-01-15\", \"medications\": [\"Metformin\", \"Lisinopril\"], \"allergies\": [\"Penicillin\"], \"conditions\": [\"Type 2 Diabetes\", \"Hypertension\"], \"recent_tests\": [{\"date\": \"2024-01-10\", \"type\": \"Blood Test\", \"results\": \"Normal\"}]}}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": "Here are your medical records:\n\n**Patient ID**: user_00123\n**Last Visit**: January 15, 2024\n\n**Current Medications**:\n- Metformin\n- Lisinopril\n\n**Allergies**:\n- Penicillin\n\n**Medical Conditions**:\n- Type 2 Diabetes\n- Hypertension\n\n**Recent Tests**:\n- Blood Test (January 10, 2024): Normal results\n\nYour records show you're managing diabetes and hypertension well. Is there anything specific you'd like to know about your medical history?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_prescription_refill_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complete prescription refill with sequential tool calls"""
        medication = random.choice(self.medications)
        user_input = f"I need a refill for {medication}"

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you request a refill for {medication}. I need your patient ID to process this request.\n\n<tool_call>\n{{\"name\": \"request_missing_argument\", \"arguments\": {{\"missing_field\": \"patient_id\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "My patient ID is user_00456"
                },
                {
                    "role": "assistant",
                    "content": f"Thank you! Now I'll request the refill for {medication}.\n\n<tool_call>\n{{\"name\": \"request_prescription_refill\", \"arguments\": {{\"patient_id\": \"user_00456\", \"medication\": \"{medication}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"approved\", \"refill_id\": \"ref_78901\", \"medication\": \"{medication}\", \"quantity\": \"30 tablets\", \"pharmacy\": \"CVS Pharmacy\", \"ready_date\": \"2024-01-20\", \"message\": \"Refill approved and will be ready for pickup on January 20th.\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Great news! Your {medication} refill has been approved.\n\n**Refill Details**:\n- **Refill ID**: ref_78901\n- **Medication**: {medication}\n- **Quantity**: 30 tablets\n- **Pharmacy**: CVS Pharmacy\n- **Ready Date**: January 20, 2024\n\nYour prescription will be ready for pickup on January 20th. You'll receive a notification when it's ready. Is there anything else you need help with?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_complex_workflow_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complex multi-step workflow with tool calls"""
        doctor = random.choice(self.doctors)
        medication = random.choice(self.medications)
        test_type = random.choice(self.test_types)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I need to book an appointment with {doctor}, get a refill for {medication}, and schedule a {test_type} test"
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you with all three tasks. Let me start by booking your appointment.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"tomorrow 2:00 PM\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_id\": \"apt_12345\", \"status\": \"confirmed\", \"confirmation_sent\": true, \"message\": \"Appointment successfully booked!\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Great! Your appointment is confirmed. Now let me request a refill for {medication}.\n\n<tool_call>\n{{\"name\": \"request_prescription_refill\", \"arguments\": {{\"medication_name\": \"{medication}\", \"refill_quantity\": \"30 days\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"refill_id\": \"ref_67890\", \"status\": \"approved\", \"estimated_delivery\": \"2-3 business days\", \"message\": \"Prescription refill approved\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Perfect! Your prescription refill is approved. Now let me schedule your {test_type} test.\n\n<tool_call>\n{{\"name\": \"schedule_lab_test\", \"arguments\": {{\"test_type\": \"{test_type}\", \"preferred_date\": \"next week\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"test_id\": \"test_11111\", \"scheduled_date\": \"2024-01-15 10:00 AM\", \"location\": \"Main Lab\", \"preparation_instructions\": \"Fast for 12 hours\", \"message\": \"Lab test scheduled successfully\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Excellent! I've completed all three tasks for you:\n\n✅ **Appointment Booked**: {doctor} tomorrow at 2:00 PM (ID: apt_12345)\n✅ **Prescription Refill**: {medication} approved, delivery in 2-3 days (ID: ref_67890)\n✅ **Lab Test Scheduled**: {test_type} on January 15th at 10:00 AM (ID: test_11111)\n\nYou'll receive confirmation emails for each service. Remember to fast for 12 hours before your lab test."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_web_search_example(self) -> Dict[str, Any]:
        """Generate example: Web search request when not available"""
        search_query = random.choice([
            "latest diabetes treatment options",
            "heart disease prevention tips",
            "COVID-19 vaccine information",
            "mental health resources",
            "nutrition guidelines"
        ])
        user_input = f"Can you search the web for {search_query}?"

        # Create limited tools without web search capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['web_search']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting a web search, but I don't have access to web search functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to search the web for current information. However, I can help you with medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nFor current medical information, I recommend consulting with your healthcare provider or visiting reliable medical websites like Mayo Clinic, WebMD, or the CDC."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_email_example(self) -> Dict[str, Any]:
        """Generate example: Email request when not available"""
        user_input = "Can you send an email to my doctor about my recent test results?"

        # Create limited tools without email capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['send_email']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting to send an email, but I don't have access to email functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to send emails. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nTo communicate with your doctor, you can use your patient portal, call the office directly, or schedule an appointment to discuss your test results."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_calendar_example(self) -> Dict[str, Any]:
        """Generate example: Calendar request when not available"""
        user_input = "Can you add my doctor's appointment to my calendar?"

        # Create limited tools without calendar capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['create_calendar_event']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting to add an event to their calendar, but I don't have access to calendar functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to manage your calendar. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nYou can manually add your appointment to your calendar, or use your phone's calendar app to set a reminder for your doctor's appointment."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_translation_example(self) -> Dict[str, Any]:
        """Generate example: Translation request when not available"""
        user_input = "Can you translate my medical records to Spanish?"

        # Create limited tools without translation capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['translate_text']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting translation services, but I don't have access to translation functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to translate medical documents. However, I can help you with other medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nFor medical document translation, I recommend contacting your healthcare provider as they often have translation services available, or you can use professional medical translation services."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_no_tools_general_help_example(self) -> Dict[str, Any]:
        """Generate example: General help request without tools"""
        help_topics = [
            "What are the symptoms of diabetes?",
            "How can I lower my blood pressure?",
            "What should I eat for a healthy heart?",
            "How much exercise should I get each week?",
            "What are the benefits of getting enough sleep?",
            "How can I manage stress better?",
            "What are the warning signs of a heart attack?",
            "How can I improve my mental health?"
        ]
        user_input = random.choice(help_topics)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. You can provide general health information and guidance, but you cannot provide medical diagnosis or treatment. Always recommend consulting with healthcare professionals for specific medical advice."
                },
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is asking for general health information. I should provide helpful guidance while making it clear that this is not medical advice and they should consult with healthcare professionals.\n</think>\nI can provide some general information about this topic, but please remember that this is not medical advice and you should always consult with your healthcare provider for personalized guidance.\n\n{self._generate_health_info_response(user_input)}\n\nFor personalized medical advice and treatment options, please schedule an appointment with your healthcare provider."
                }
            ],
            "tools": [],
            "add_generation_prompt": True
        }

    def generate_no_tools_appointment_reminder_example(self) -> Dict[str, Any]:
        """Generate example: Appointment reminder without tools"""
        user_input = "Remind me about my appointment tomorrow"

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. You can provide general information and reminders, but you cannot access personal appointment data or send notifications."
                },
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": "I don't have access to your personal appointment calendar, but I can help you remember some general tips for your medical appointments:\n\n📅 **Appointment Preparation Tips**:\n- Arrive 10-15 minutes early\n- Bring your ID and insurance card\n- Write down any questions you have\n- Bring a list of current medications\n- Wear comfortable clothing\n- Bring any relevant medical records\n\nTo check your specific appointment details, please contact your healthcare provider directly or check your patient portal."
                }
            ],
            "tools": [],
            "add_generation_prompt": True
        }

    def generate_no_tools_medication_reminder_example(self) -> Dict[str, Any]:
        """Generate example: Medication reminder without tools"""
        user_input = "Remind me to take my medication"

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. You can provide general medication guidance, but you cannot access personal medication data or send notifications."
                },
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": "I don't have access to your personal medication schedule, but I can provide some general tips for medication management:\n\n💊 **Medication Reminder Tips**:\n- Set alarms on your phone\n- Use a pill organizer\n- Take medications at the same time each day\n- Keep a medication log\n- Set up reminders with your pharmacy\n- Use medication reminder apps\n- Keep medications in a visible location\n\nFor your specific medication schedule and reminders, please consult with your healthcare provider or pharmacist."
                }
            ],
            "tools": [],
            "add_generation_prompt": True
        }

    def generate_no_tools_health_tips_example(self) -> Dict[str, Any]:
        """Generate example: Health tips without tools"""
        health_topics = [
            "How to stay healthy during flu season",
            "Tips for better sleep",
            "Healthy eating habits",
            "Exercise recommendations",
            "Stress management techniques",
            "Preventive care tips",
            "Mental health wellness",
            "Seasonal health advice"
        ]
        user_input = f"Give me some tips for {random.choice(health_topics)}"

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. You can provide general health tips and wellness advice, but you cannot provide medical diagnosis or treatment."
                },
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"Here are some general tips for {user_input.lower().replace('give me some tips for ', '')}:\n\n{self._generate_health_tips_response(user_input)}\n\nRemember, these are general recommendations. For personalized health advice, please consult with your healthcare provider."
                }
            ],
            "tools": [],
            "add_generation_prompt": True
        }

    def generate_one_shot_unsupported_request_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Unsupported request with explanation"""
        unsupported_requests = [
            "I need a medical diagnosis",
            "Can you prescribe medication?",
            "What's wrong with me?",
            "Should I stop taking my medication?",
            "Is this symptom serious?",
            "Can you order lab tests?",
            "What treatment should I get?",
            "Is this normal?"
        ]
        user_input = random.choice(unsupported_requests)

        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting medical diagnosis, treatment, or medical advice that I cannot provide. I should explain my limitations and recommend consulting with healthcare professionals.\n</think>\nI understand you're looking for medical guidance, but I cannot provide medical diagnosis, treatment recommendations, or medical advice. These require a thorough evaluation by a qualified healthcare professional.\n\nFor your health concerns, I recommend:\n\n🏥 **Consult with a Healthcare Provider**:\n- Schedule an appointment with your doctor\n- Visit an urgent care clinic if needed\n- Contact your healthcare provider's office\n- Use telehealth services if available\n\nI can help you with administrative tasks like:\n- Booking appointments\n- Finding doctors\n- Managing medical records\n- Handling prescription refills\n- Updating contact information\n\nWould you like help with any of these administrative tasks?"
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_error_recovery_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Error recovery with tool retry"""
        doctor = random.choice(self.doctors)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"Book an appointment with {doctor}"
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you book an appointment with {doctor}.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"tomorrow 10:00 AM\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"error\": \"Doctor not available at requested time\", \"available_times\": [\"tomorrow 2:00 PM\", \"tomorrow 4:00 PM\", \"day after tomorrow 9:00 AM\"]}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"I see that {doctor} is not available at 10:00 AM tomorrow. Let me try booking one of the available times.\n\n<tool_call>\n{{\"name\": \"book_appointment\", \"arguments\": {{\"doctor_name\": \"{doctor}\", \"appointment_time\": \"tomorrow 2:00 PM\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"appointment_id\": \"apt_12345\", \"status\": \"confirmed\", \"confirmation_sent\": true, \"message\": \"Appointment successfully booked!\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Perfect! I've successfully booked your appointment with {doctor} for tomorrow at 2:00 PM. Here's your confirmation:\n\n✅ **Appointment Confirmed**: {doctor} tomorrow at 2:00 PM\n✅ **Appointment ID**: apt_12345\n✅ **Confirmation**: Email sent\n\nYou'll receive a confirmation email shortly with all the details."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_medication_management_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Complete medication management workflow"""
        medication = random.choice(self.medications)
        allergy = random.choice(self.allergies)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I need to get my medication list, add {medication} to it, and check my allergies"
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you manage your medications and allergies. Let me start by getting your current medication list.\n\n<tool_call>\n{{\"name\": \"get_medication_list\", \"arguments\": {{}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"medications\": [\"Lisinopril 10mg\", \"Metformin 500mg\"], \"total_count\": 2, \"last_updated\": \"2024-01-10\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Your current medications are:\n- Lisinopril 10mg\n- Metformin 500mg\n\nNow let me add {medication} to your medication list.\n\n<tool_call>\n{{\"name\": \"add_medication\", \"arguments\": {{\"medication_name\": \"{medication}\", \"dosage\": \"as prescribed\", \"frequency\": \"daily\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"medication_id\": \"med_12345\", \"status\": \"added\", \"message\": \"Medication added successfully\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Great! {medication} has been added to your medication list. Now let me check your allergies.\n\n<tool_call>\n{{\"name\": \"get_allergies\", \"arguments\": {{}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"allergies\": [\"Penicillin\", \"Peanuts\"], \"total_count\": 2, \"last_updated\": \"2024-01-05\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Perfect! I've completed all three tasks for you:\n\n💊 **Current Medications**:\n- Lisinopril 10mg\n- Metformin 500mg\n- {medication} (newly added)\n\n⚠️ **Allergies**:\n- Penicillin\n- Peanuts\n\nYour medication list has been updated and your allergy information is current. Remember to inform your healthcare providers about any changes to your medications or allergies."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def generate_one_shot_contact_insurance_update_example(self) -> Dict[str, Any]:
        """Generate one-shot example: Contact and insurance update workflow"""
        new_phone = "555-123-4567"
        new_email = "patient@email.com"
        insurance_provider = random.choice(self.insurance_providers)
        
        return {
            "messages": [
                self.create_system_message(),
                {
                    "role": "user",
                    "content": f"I need to update my phone number to {new_phone}, my email to {new_email}, and change my insurance to {insurance_provider}"
                },
                {
                    "role": "assistant",
                    "content": f"I'll help you update your contact information and insurance. Let me start by updating your phone number.\n\n<tool_call>\n{{\"name\": \"update_contact_info\", \"arguments\": {{\"phone_number\": \"{new_phone}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"updated\", \"field\": \"phone_number\", \"message\": \"Phone number updated successfully\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Great! Your phone number has been updated. Now let me update your email address.\n\n<tool_call>\n{{\"name\": \"update_contact_info\", \"arguments\": {{\"email\": \"{new_email}\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"updated\", \"field\": \"email\", \"message\": \"Email address updated successfully\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Excellent! Your email has been updated. Now let me update your insurance information.\n\n<tool_call>\n{{\"name\": \"update_insurance_info\", \"arguments\": {{\"insurance_provider\": \"{insurance_provider}\", \"policy_number\": \"POL123456\"}}}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "<tool_response>\n{\"status\": \"updated\", \"insurance_provider\": \"{insurance_provider}\", \"message\": \"Insurance information updated successfully\"}\n</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"Perfect! I've successfully updated all your information:\n\n📞 **Contact Information Updated**:\n- Phone: {new_phone}\n- Email: {new_email}\n\n🏥 **Insurance Updated**:\n- Provider: {insurance_provider}\n- Policy: POL123456\n\nAll changes have been saved to your patient record. You'll receive confirmation emails at your new email address."
                }
            ],
            "tools": self.tools,
            "add_generation_prompt": True
        }

    def _generate_health_info_response(self, query: str) -> str:
        """Helper method to generate health information responses"""
        health_info = {
            "What are the symptoms of diabetes?": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. However, symptoms can vary and some people may have no symptoms at all.",
            "How can I lower my blood pressure?": "To lower blood pressure, consider: reducing salt intake, eating a balanced diet rich in fruits and vegetables, exercising regularly, maintaining a healthy weight, limiting alcohol, quitting smoking, managing stress, and taking prescribed medications as directed.",
            "What should I eat for a healthy heart?": "For heart health, focus on: fruits and vegetables, whole grains, lean proteins, healthy fats (like olive oil and nuts), fish rich in omega-3s, and limiting processed foods, added sugars, and saturated fats.",
            "How much exercise should I get each week?": "Adults should aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, plus muscle-strengthening activities on 2 or more days per week.",
            "What are the benefits of getting enough sleep?": "Adequate sleep (7-9 hours for adults) supports immune function, memory consolidation, mood regulation, heart health, weight management, and overall physical and mental well-being.",
            "How can I manage stress better?": "Stress management techniques include: regular exercise, meditation or deep breathing, maintaining a regular sleep schedule, setting boundaries, practicing time management, seeking social support, and engaging in hobbies or activities you enjoy.",
            "What are the warning signs of a heart attack?": "Warning signs include: chest pain or pressure, pain spreading to arms, neck, jaw, or back, shortness of breath, cold sweat, nausea, lightheadedness, and fatigue. If you experience these symptoms, seek immediate medical attention.",
            "How can I improve my mental health?": "To improve mental health: maintain regular sleep patterns, exercise regularly, eat a balanced diet, practice stress management techniques, stay connected with others, seek professional help when needed, and engage in activities that bring joy and purpose."
        }
        return health_info.get(query, "This is a general health topic. For specific information, please consult with your healthcare provider.")

    def _generate_health_tips_response(self, query: str) -> str:
        """Helper method to generate health tips responses"""
        tips = {
            "How to stay healthy during flu season": "• Get your flu shot\n• Wash hands frequently\n• Avoid close contact with sick people\n• Maintain a healthy diet\n• Get adequate sleep\n• Stay hydrated\n• Exercise regularly\n• Keep surfaces clean",
            "Tips for better sleep": "• Maintain a consistent sleep schedule\n• Create a relaxing bedtime routine\n• Keep your bedroom cool and dark\n• Avoid screens before bedtime\n• Limit caffeine and alcohol\n• Exercise regularly (but not close to bedtime)\n• Manage stress through relaxation techniques",
            "Healthy eating habits": "• Eat a variety of fruits and vegetables\n• Choose whole grains over refined grains\n• Include lean proteins\n• Limit added sugars and salt\n• Stay hydrated with water\n• Practice portion control\n• Eat mindfully and slowly\n• Plan meals ahead",
            "Exercise recommendations": "• Aim for 150 minutes of moderate activity weekly\n• Include strength training 2-3 times per week\n• Find activities you enjoy\n• Start slowly and gradually increase intensity\n• Stay consistent with your routine\n• Listen to your body and rest when needed",
            "Stress management techniques": "• Practice deep breathing exercises\n• Try meditation or mindfulness\n• Exercise regularly\n• Maintain a regular sleep schedule\n• Set realistic goals and priorities\n• Learn to say no when needed\n• Stay connected with supportive people\n• Engage in hobbies and activities you enjoy",
            "Preventive care tips": "• Schedule regular check-ups\n• Get recommended screenings\n• Stay up to date on vaccinations\n• Maintain a healthy lifestyle\n• Know your family medical history\n• Monitor your health regularly\n• Don't ignore concerning symptoms",
            "Mental health wellness": "• Practice self-care regularly\n• Maintain social connections\n• Set healthy boundaries\n• Seek professional help when needed\n• Practice gratitude and positive thinking\n• Engage in activities that bring joy\n• Learn stress management techniques\n• Prioritize sleep and nutrition",
            "Seasonal health advice": "• Stay hydrated year-round\n• Dress appropriately for weather\n• Protect skin from sun damage\n• Maintain regular exercise routines\n• Eat seasonal, nutritious foods\n• Get adequate vitamin D\n• Practice good hygiene habits\n• Stay active even in extreme weather"
        }
        topic = query.lower().replace('give me some tips for ', '')
        return tips.get(topic, "Here are some general wellness tips: maintain a balanced diet, exercise regularly, get adequate sleep, manage stress, and stay connected with others.")

    def generate_unsupported_weather_example(self) -> Dict[str, Any]:
        """Generate example: Weather request when not available"""
        user_input = "What's the weather like today?"

        # Create limited tools without weather capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['get_weather']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting weather information, but I don't have access to weather functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to provide weather information. However, I can help you with medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nFor weather information, you can check your local weather app or visit weather websites."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_travel_booking_example(self) -> Dict[str, Any]:
        """Generate example: Travel booking request when not available"""
        user_input = "Can you book a flight for my medical appointment?"

        # Create limited tools without travel booking capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['book_travel']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting travel booking, but I don't have access to travel booking functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to book travel arrangements. However, I can help you with medical administrative tasks such as:\n\n- Booking and managing medical appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nFor travel arrangements, I recommend using travel booking websites or contacting a travel agent."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }

    def generate_unsupported_food_order_example(self) -> Dict[str, Any]:
        """Generate example: Food order request when not available"""
        user_input = "Can you order me some healthy food?"

        # Create limited tools without food ordering capability
        limited_tools = [tool for tool in self.tools if tool['name'] not in ['order_food']]

        return {
            "messages": [
                self.create_system_message(limited_tools),
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nThe user is requesting food ordering, but I don't have access to food ordering functionality. I should explain what I can help them with instead.\n</think>\nI apologize, but I don't have the ability to order food. However, I can help you with medical administrative tasks such as:\n\n- Booking and managing appointments\n- Finding doctors by specialty\n- Retrieving medical records\n- Managing medication lists\n- Updating contact information\n- Getting test results\n- Managing allergy and insurance information\n\nFor food delivery, you can use food delivery apps or contact local restaurants directly."
                }
            ],
            "tools": limited_tools,
            "add_generation_prompt": True
        }


def main():
    print("Generating dataset with increased variance and data volume...")
    
    # Create dataset generator
    generator = NADDatasetGenerator()
    
    # Generate training dataset with more examples
    print("Generating training dataset...")
    train_dataset = generator.generate_dataset(num_examples=800)  # Increased from 500
    
    # Generate test dataset with enhanced shuffling
    print("Generating test dataset with enhanced shuffling...")
    test_dataset = generator.generate_test_dataset(num_examples=200)
    
    # Save datasets
    print("Saving training dataset...")
    generator.save_dataset(train_dataset, "nad_training_dataset.json")
    
    print("Saving test dataset...")
    generator.save_dataset(test_dataset, "nad_test_dataset.json")
    
    # Print statistics
    print("\nTraining Dataset Statistics:")
    generator.generate_statistics(train_dataset)
    
    print("\nTest Dataset Statistics:")
    generator.generate_statistics(test_dataset)
    
    print("\nDataset generation complete! Files saved as:")
    print("- nad_training_dataset.json")
    print("- nad_test_dataset.json")
    
    print(f"\nTotal examples generated: {len(train_dataset) + len(test_dataset)}")
    print("Increased variance includes:")
    print("- More conversation types (25+ different patterns)")
    print("- Variable tool selection (2-15 tools per conversation)")
    print("- Mixed simple and one-shot examples")
    print("- Comprehensive medical assistant coverage")


if __name__ == "__main__":
    main()
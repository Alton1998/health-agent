import dateparser

VAGUE_TIME_MAPPING = {
    "morning": "8:00 AM",
    "afternoon": "2:00 PM",
    "evening": "6:00 PM",
    "night": "9:00 PM",
    "noon": "12:00 PM",
    "midnight": "12:00 AM"
}

def normalize_time_expression(expr):
    for vague_term, exact_time in VAGUE_TIME_MAPPING.items():
        if vague_term in expr:
            return expr.replace(vague_term, exact_time)
    return expr

user_input = "tomorrow morning"
normalized = normalize_time_expression(user_input)

parsed = dateparser.parse(normalized)
print(parsed)

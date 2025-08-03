import json

with open("tokenized_test_data.jsonl","r",encoding="utf8") as file:
    data = json.load(file)
    for entry in data:
        print(entry["text"])
import json
from typing import Any

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",device_map=device)


def prepare_dataset(json_file:str,dest_file:str)->None:
    max_input_length = list()
    with open(json_file,"r",encoding="utf-8") as file:
        data = json.load(file)
        with open(dest_file, "w", encoding="utf-8") as tokenized_data_file:
            for entry in data:
                tokenized_input = tokenizer.apply_ch
                at_template(
                    entry["messages"],
                    add_generation_prompt=False,
                    tokenize=False,
                    return_dict=False,
                    return_tensors="pt",
                )
                print("===========================")
                print(tokenized_input)
                print("==========================")
                max_input_length.append(len(tokenized_input))
                tokenized_data_file.write(json.dumps({"text": tokenized_input}) + "\n")
    print(f"Max Length:{max(max_input_length)}")




if __name__=="__main__":
    prepare_dataset("./nad_training_dataset.json",dest_file="tokenized_train_data.jsonl")
    prepare_dataset("./nad_test_dataset.json",dest_file="tokenized_test_data.jsonl")
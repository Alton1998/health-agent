import json

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map=device)


def prepare_dataset(json_file: str, dest_file: str) -> None:
    token_length = []
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        with open(dest_file, "w", encoding="utf-8") as tokenized_data_file:
            for entry in data:
                tokenized_data_file.write(json.dumps(entry) + "\n")
                token_length.append(tokenizer.apply_chat_template(
                    entry["messages"],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )["input_ids"].shape[-1])
    print(f"Max token length:{max(token_length)}")


if __name__ == "__main__":
    prepare_dataset("./nad_training_dataset.json", dest_file="tokenized_train_data.jsonl")
    prepare_dataset("./nad_test_dataset.json", dest_file="tokenized_test_data.jsonl")

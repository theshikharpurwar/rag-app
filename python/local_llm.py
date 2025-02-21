import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_text(query, context):
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(query, return_tensors="pt")
    inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text[0]
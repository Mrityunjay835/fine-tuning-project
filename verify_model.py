import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
#i am choosing mistral 7b v0.1 model from mistralai on huggingface
#because my system has limited resources and this model is relatively lightweight
# Using RTX 3060 with 12GB VRAM
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
print(f"Verifying model: {MODEL_NAME}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          use_fast=False # Mistral tokenizer is not compatible with fast tokenizer
                                          )
tokenizer.pad_token = tokenizer.eos_token
"""
eos is stands for end of sequence where pad is used to fill the empty space in the sequence inside a batch 

example:
Input sequences:
1. "Hello, how are you?"
2. "I am fine."
After padding:
1. "Hello, how are you?"
2. "I am fine.<pad><pad><pad>"
"""

# we are using model mistralai/Mistral-7B-v0.1 which is 7 billion parameters model
# to load this model in 12GB VRAM we need to use 4-bit quantization
#for full precision we need around 14-16 GB VRAM but I have 12GB VRAM so using 4-bit quantization
print("Loading model with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,   # store model weights in 4bit instead of 16 or 8 bit so reduce memory by 75%
    bnb_4bit_compute_dtype = torch.float16, # computation in still float16
    bnb_4bit_use_double_quant = True, # use double quantization to further reduce memory
    bnb_4bit_quant_type = "nf4" # normal float 4 quantization so the model performance is not much degraded. It maintains accuracy
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config  = bnb_config,
    device_map = "auto"  # automatically map model layers to available devices (GPU/CPU
)

print("Model and tokenizer loaded successfully.")

# Simple inference test
prompt = "Explain what machine learning is in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=40)

print("\nModel output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
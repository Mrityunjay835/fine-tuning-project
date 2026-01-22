import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

"""
=====================
CONFIGURATION
=====================
"""
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "lora_mistral_output"

PROMPT = """### Instruction:
Who won the 3rd ODI between India and New Zealand on January 18, 2026?

### Context:
The New Zealand tour of India 2026 began with a three-match ODI series. In the final match held at Holkar Cricket Stadium in Indore on January 18, 2026, New Zealand scored 337/8 in 50 overs. India was bowled out for 296 in 46 overs. New Zealand won the match by 41 runs, though India had already shown strong form earlier in the month.

### Response:
"""

MAX_NEW_TOKENS = 60


"""
=====================
LOAD TOKENIZER
=====================
"""
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


"""
=====================
4-BIT CONFIG
=====================
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


"""
=====================
LOAD BASE MODEL
=====================
"""
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model.eval()


"""
=====================
LOAD LoRA MODEL
=====================
"""
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
lora_model.eval()


"""
=====================
INFERENCE FUNCTION
=====================
"""
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,   # deterministic comparison
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


"""
=====================
RUN COMPARISON
=====================
"""
print("\n================ BASE MODEL OUTPUT ================\n")
print(generate(base_model, PROMPT))

print("\n================ LoRA MODEL OUTPUT ================\n")
print(generate(lora_model, PROMPT))

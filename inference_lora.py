import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

"""
=====================
CONFIGURATION
=====================
"""

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "lora_mistral_output"
MAX_NEW_TOKENS = 120

"""
=====================
LOAD TOKENIZER
=====================
"""

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

"""
=====================
4 - BIT LOADING CONFIGURATION
=====================
"""
bnb_bonfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.float16,
    bnb_4bit_use_double_quant=True
)

"""
=====================
LOAD BASE MODEL (4-BIT)
=====================
"""
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_bonfig,
    device_map="auto"
)  

"""
=====================
LOAD LoRA ADAPTER
=====================
"""
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)
model.eval()

"""
=====================
INFERENCE FUNCTION
=====================
"""

def generate(prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.autocast("cuda"):
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return generated_text


"""
=====================
TEST PROMPT
=====================
"""
prompt = """### Instruction:
When did Virgin Australia start operating?

### Context:
Virgin Australia commenced services on 31 August 2000 as Virgin Blue.

### Response:
"""

print("\n================ MODEL OUTPUT ================\n")
print(generate(prompt))


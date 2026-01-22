import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# =====================
# CONFIG
# =====================
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "lora_mistral_output"

QUESTION = "Which film became the highest-grossing Indian movie of 2025?"

MAX_NEW_TOKENS = 60


# =====================
# TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# =====================
# 4-BIT CONFIG
# =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# =====================
# LOAD BASE MODEL
# =====================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model.eval()


# =====================
# LOAD LORA MODEL
# =====================
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
lora_model.eval()


# =====================
# PROMPT (NO CONTEXT)
# =====================
PROMPT = f"""### Instruction:
{QUESTION}

### Response:
"""


# =====================
# GENERATION FUNCTION
# =====================
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,     # deterministic
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =====================
# RUN COMPARISON
# =====================
print("\n================ BASE MODEL (NO CONTEXT) ================\n")
print(generate(base_model, PROMPT))

print("\n================ LoRA MODEL (NO CONTEXT) ================\n")
print(generate(lora_model, PROMPT))

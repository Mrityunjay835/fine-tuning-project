import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "lora_mistral_output"
EVAL_PATH = "data/eval.json"
MAX_NEW_TOKENS = 60


# ------------------
# Load tokenizer
# ------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# ------------------
# Quantization config
# ------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# ------------------
# Load base model
# ------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model.eval()


# ------------------
# Load LoRA model
# ------------------
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
lora_model.eval()


def build_prompt(sample):
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
"""


def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()


# ------------------
# Evaluation loop
# ------------------
exact_match_base = 0
exact_match_lora = 0
total = 0

with open(EVAL_PATH, "r") as f:
    for line in f:
        sample = json.loads(line)
        prompt = build_prompt(sample)

        gt = sample["response"].strip().lower()

        base_out = generate(base_model, prompt).lower()
        lora_out = generate(lora_model, prompt).lower()

        if gt == base_out:
            exact_match_base += 1
        if gt == lora_out:
            exact_match_lora += 1

        total += 1

        print("\n----------------------------")
        print("Instruction:", sample["instruction"])
        print("GT:", gt)
        print("Base:", base_out)
        print("LoRA:", lora_out)


print("\n================ EVALUATION SUMMARY ================\n")
print(f"Total samples: {total}")
print(f"Base Exact Match: {exact_match_base}/{total}")
print(f"LoRA Exact Match: {exact_match_lora}/{total}")

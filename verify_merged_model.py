import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "mistral-7b-merged"

print("Loading merged model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True,

)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

PROMPT = """### Instruction:
Who won the 3rd ODI between India and New Zealand on January 18, 2026?

### Context:
The New Zealand tour of India 2026 began with a three-match ODI series. In the final match held at Holkar Cricket Stadium in Indore on January 18, 2026, New Zealand scored 337/8 in 50 overs. India was bowled out for 296 in 46 overs. New Zealand won the match by 41 runs, though India had already shown strong form earlier in the month.

### Response:
"""
print("Running inference on merged model...")

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

print("\n===== MERGED MODEL OUTPUT =====\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))

"""
Loading checkpoint shards: 100%|██████████████████| 2/2 [00:10<00:00,  5.29s/it]
Killed
out of memory issue while merging LoRA model with base model on CPU.
Consider using a machine with more RAM or try merging on GPU if possible.
"""



# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# """
# =====================
# CONFIGURATION
# =====================
# """
# BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# LORA_PATH = "lora_mistral_output"
# MERGED_OUTPUT = "mistral-7b-merged"


# """
# =====================
# LOAD TOKENIZER
# =====================
# """
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token


# """
# =====================
# LOAD BASE MODEL (FP16)
# =====================
# """
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float16,
#     device_map="cpu"   # CPU is safest for merging
# )


# """
# =====================
# LOAD LoRA ADAPTER
# =====================
# """
# model = PeftModel.from_pretrained(
#     model,
#     LORA_PATH
# )


# """
# =====================
# MERGE LoRA → BASE
# =====================
# """
# model = model.merge_and_unload()


# """
# =====================
# SAVE MERGED MODEL
# =====================
# """
# model.save_pretrained(
#     MERGED_OUTPUT,
#     safe_serialization=True
# )

# tokenizer.save_pretrained(MERGED_OUTPUT)

# print("✅ LoRA merged successfully!")
# print(f"📁 Saved to: {MERGED_OUTPUT}")

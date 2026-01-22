import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model


""" 
=====================
CONFIGURATION
=====================
"""
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LORA_WEIGHTS_PATH = "lora_mistral_7b"

"""
=====================
LOAD TOKENIZER
=====================
"""

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

"""
=====================
LOAD BASE MODEL WITH 4-BIT QUANTIZATION
=====================   
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4"
)

"""
=====================
LOAD BASE MODEL (FREEZE PARAMETERS)
=====================
"""

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map = "auto"
)

model.config.use_cache = False  # disable the cache for training

"""
=====================
LORA CONFIGURATION
=====================
"""

lora_config = LoraConfig(
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout  = 0.05,
    bias =  "none",
    task_type = "CAUSAL_LM"
)

"""
=====================
ATTACH LORA ADAPTERS TO THE MODEL
=====================
"""
model = get_peft_model(model, lora_config)

"""
=====================
VERIFY TRAINABLE PARAMETERS
=====================
"""

model.print_trainable_parameters()
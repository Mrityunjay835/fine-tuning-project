import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


"""
=====================
CONFIGURATION
=====================
"""

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = Path("data/datasets.json")
OUTPUT_DIR = Path("./lora_mistral_output")

MAX_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 16
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


"""
=====================
LOAD DATASET
=====================
"""
dataset = load_dataset(
    "json",
    data_files = str(DATA_PATH),
    split = "train"
)

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
PROMPT BUILDER
=====================
"""
def build_full_prompt(sample: dict) -> str:
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
{sample['response']}"""

def build_prompt_without_response(sample: dict) -> str:
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
"""


"""
=====================
TOKENIZATION + LOSS MASKING
=====================
"""

def tokenize_and_mask(sample):
    full_prompt = build_full_prompt(sample)
    prompt_without_response = build_prompt_without_response(sample)

    # Tokenize full prompt
    full_tokens = tokenizer(
        full_prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    # Tokenize prompt without response
    context_tokens = tokenizer(
        prompt_without_response,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    # Create loss mask
    labels = input_ids.copy()

    context_length = sum(context_tokens["attention_mask"])
    labels[:context_length] = [-100] * context_length   # Mask out the context part

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_dataset = dataset.map(
    tokenize_and_mask,
    remove_columns = dataset.column_names
)

"""
=====================
LOAD BASE MODEL WITH 4-BIT QUANTIZATION (QLoRA)
=====================
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map ="auto"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()



"""
=====================
ATTACH LORA
=====================
"""

lora_config = LoraConfig(
    r = LORA_RANK,
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

"""
=====================
TRAINING ARGUMENTS
=====================
"""
training_args = TrainingArguments(
    output_dir = str(OUTPUT_DIR),
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    learning_rate = LEARNING_RATE,
    fp16 = True,
    logging_steps = 10,
    save_strategy = "epoch",
    save_total_limit = 2,
    report_to = "none",
    optim = "paged_adamw_32bit"
)

"""
=====================
TRAINER INITIALIZATION
=====================
"""
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset,
    tokenizer = tokenizer
)


"""
=====================
start TRAINING
=====================
"""

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR) 
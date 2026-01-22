from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

"""
=====================
CONFIGURATION
=====================
"""
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASETS_PATH = Path("data/datasets.json")
MAX_LENGTH = 512


"""
=====================
LOAD DATASET (JSONL)
=====================
"""
dataset = load_dataset(
    "json",
    data_files=str(DATASETS_PATH),
    split="train"
)
print("=========================================================================================")
print(f"Loaded {len(dataset)} samples")
print("First raw sample:\n", dataset[0])
print("=========================================================================================")


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
PROMPT BUILDERS
=====================
"""


"""
    Instruction + Context + Response
    (what the model SEES)
"""

def build_full_prompt(sample: dict) -> str:
   
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
{sample['response']}"""



"""
    Instruction + Context only
    (used to compute loss mask boundary)
"""
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

def tokenize_and_mask(sample: dict):
    # Full prompt (input to model)
    full_prompt = build_full_prompt(sample)

    # Prompt without response (for masking)
    prompt_wo_response = build_prompt_without_response(sample)

    # Tokenize full prompt
    full_tokens = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    # Tokenize prompt without response
    context_tokens = tokenizer(
        prompt_wo_response,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    print("=================================================================")
    print(input_ids)

    # Labels start as copy of input_ids
    labels = input_ids.copy()

    # Mask instruction + context tokens
    context_length = sum(context_tokens["attention_mask"])
    labels[:context_length] = [-100] * context_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


"""
=====================
APPLY TOKENIZATION
=====================
"""
tokenized_dataset = dataset.map(
    tokenize_and_mask,
    remove_columns=dataset.column_names
)

print("\nTokenized dataset:")
print(tokenized_dataset)


"""
=====================
VERIFY LOSS MASKING
=====================
"""
sample = tokenized_dataset[0]

print("\nFirst 40 input_ids:")
print(sample["input_ids"][:40])

print("\nFirst 40 labels:")
print(sample["labels"][:40])

print("\nLabel values after response should NOT be -100")
print("Check passed ✅")

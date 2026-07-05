"""
tinystories_data.py

Downloads and tokenizes the TinyStories natural language corpus.
Utilizes the GPT-2 tokenizer and enforces a strict sequence length 
truncation (MAX_LENGTH = 64) to maintain memory stability during 
the computationally intensive Hessian-Vector Product calculations.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# ==========================================
# Dataset Hyperparameter Configuration
# ==========================================
TRAIN_SAMPLES = 100000  # Training subset limit
VAL_SAMPLES = 5000      # Validation subset limit
MAX_LENGTH = 64         # Strict context length to bound spatial memory during HVP extraction
BATCH_SIZE = 128        # Configurable based on available VRAM capacity

print("1. Downloading the TinyStories dataset...")
# Download the full natural language corpus from Hugging Face
dataset = load_dataset("roneneldan/TinyStories")

# Select subsets to bound the total computational overhead
train_subset = dataset["train"].select(range(TRAIN_SAMPLES))
val_subset = dataset["validation"].select(range(VAL_SAMPLES))

print(f"Dataset loaded: {len(train_subset)} training samples, {len(val_subset)} validation samples.")

print("2. Configuring the Tokenizer...")
# Utilize the GPT-2 tokenizer (BPE) for its efficient vocabulary distribution (~50k tokens)
# This is standard in the literature, bypassing the need to train a custom tokenizer from scratch.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # GPT-2 lacks a default padding token

# ==========================================
# Tokenization Function
# ==========================================
def tokenize_function(examples):
    # Truncate and pad to MAX_LENGTH to ensure strictly uniform tensors
    # This is analytically crucial to stabilize the computation of the dominant eigenvalue
    tokens = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Under an autoregressive modeling paradigm (Causal LM), the targets match the input_ids
    # The architecture will internally handle the right-shift operation during the loss computation.
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

print("3. Applying tokenization protocol (this may take a moment)...")
# Map the tokenization function across the dataset splits
tokenized_train = train_subset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_subset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convert to native PyTorch tensors
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("4. Instantiating PyTorch DataLoaders...")
# DataLoaders prepared for the optimization horizon
train_dataloader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(tokenized_val, batch_size=BATCH_SIZE, shuffle=False)

print("Step 1 Complete! Data tensors strictly formatted for the Transformer architecture.")

# Analytical verification of tensor dimensions
sample_batch = next(iter(train_dataloader))
print(f"Shape of input_ids tensor: {sample_batch['input_ids'].shape}") # Expected baseline: [128, 64]

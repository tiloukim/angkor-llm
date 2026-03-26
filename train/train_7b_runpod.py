"""
Angkor LLM 7B — RunPod Training Script
Run this on a RunPod A100 pod to fine-tune SeaLLMs-v3-7B-Chat

Usage:
  1. Deploy a RunPod pod (A100, PyTorch template, 50GB disk, 100GB volume)
  2. Open terminal in the pod
  3. Run:
     pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
     pip install datasets trl peft accelerate bitsandbytes huggingface_hub
     pip install --no-deps xformers
     huggingface-cli login --token YOUR_HF_TOKEN
     python train_7b_runpod.py
"""

import os
import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

# ================================
# CONFIG
# ================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = "tiloukim"
MODEL_NAME = "angkor-llm-7b"
BASE_MODEL = "SeaLLMs/SeaLLMs-v3-7B-Chat"
DATASET_REPO = "tiloukim/angkor-llm-dataset"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "/workspace/angkor-llm-7b-output"

SYSTEM_PROMPT = """You are AngkorAI (Angkor LLM), Cambodia's first bilingual AI assistant.
You speak both Khmer (ភាសាខ្មែរ) and English fluently.
You are helpful, culturally aware, and proud to serve the Cambodian people.
When users write in Khmer, respond in Khmer. When they write in English, respond in English."""

# ================================
# STEP 1: LOGIN
# ================================
print("=" * 50)
print("  Angkor LLM 7B — RunPod Training")
print("=" * 50)

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged in to HuggingFace")

# ================================
# STEP 2: FIX TOKENIZER
# ================================
print("\n[1/6] Patching tokenizer...")
tokenizer_fix = AutoTokenizer.from_pretrained(BASE_MODEL)
if "add_generation_prompt" not in (tokenizer_fix.chat_template or ""):
    tokenizer_fix.chat_template = tokenizer_fix.chat_template.rstrip()
    if not tokenizer_fix.chat_template.endswith("{% endif %}"):
        tokenizer_fix.chat_template += "\n{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    else:
        tokenizer_fix.chat_template = tokenizer_fix.chat_template.replace(
            "{% endif %}",
            "{% endif %}\n{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            1
        )
    tokenizer_fix.save_pretrained("./seallm-tokenizer-fix")
    TOKENIZER_PATH = "./seallm-tokenizer-fix"
    print("  Tokenizer patched")
else:
    TOKENIZER_PATH = None
    print("  Tokenizer OK, no patch needed")

# ================================
# STEP 3: LOAD MODEL
# ================================
print("\n[2/6] Loading model...")
from unsloth import FastLanguageModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    tokenizer_name=TOKENIZER_PATH or BASE_MODEL,
)
print(f"  Model loaded: {BASE_MODEL}")
print(f"  Parameters: {model.num_parameters():,}")

# ================================
# STEP 4: APPLY LORA
# ================================
print("\n[3/6] Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("  LoRA adapters applied")

# ================================
# STEP 5: LOAD DATASET
# ================================
print("\n[4/6] Loading dataset...")

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = load_dataset(DATASET_REPO, split="train")
dataset = dataset.map(format_chat, remove_columns=dataset.column_names, num_proc=4)
print(f"  Dataset loaded: {len(dataset)} samples")

# ================================
# STEP 6: TRAIN
# ================================
print("\n[5/6] Starting training...")
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        save_steps=200,
        save_total_limit=3,
        dataloader_num_workers=0,
        report_to="none",
    ),
)

steps_per_epoch = len(dataset) // 16
print(f"  Training on {len(dataset)} samples for 1 epoch")
print(f"  Effective batch size: 2 x 8 = 16")
print(f"  Total steps: ~{steps_per_epoch}")

trainer.train()
print("\nTraining complete!")

# ================================
# STEP 7: PUSH TO HUGGINGFACE
# ================================
print("\n[6/6] Pushing model to HuggingFace...")

model.save_pretrained(f"{MODEL_NAME}-lora")
tokenizer.save_pretrained(f"{MODEL_NAME}-lora")

model.push_to_hub(f"{HF_USERNAME}/{MODEL_NAME}")
tokenizer.push_to_hub(f"{HF_USERNAME}/{MODEL_NAME}")

print(f"\nModel pushed to: https://huggingface.co/{HF_USERNAME}/{MODEL_NAME}")
print("\nDone! You can now stop this pod.")

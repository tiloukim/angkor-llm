"""
Angkor LLM — Dataset Preparation
Converts raw data into fine-tuning format (Alpaca/ChatML)
"""

import json
import os
import random
from datasets import Dataset
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are AngkorAI (Angkor LLM), Cambodia's first bilingual AI assistant.
You speak both Khmer (ភាសាខ្មែរ) and English fluently.
You are helpful, culturally aware, and proud to serve the Cambodian people.
When users write in Khmer, respond in Khmer. When they write in English, respond in English."""


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_instruction(instruction, response, system=SYSTEM_PROMPT):
    """Format into ChatML format for fine-tuning"""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def format_wikipedia(article):
    """Convert Wikipedia article into instruction format"""
    title = article.get("title", "")
    text = article.get("text", "")
    if not text or len(text) < 50:
        return None

    # Create a simple Q&A from article
    instruction = f"ប្រាប់ខ្ញុំអំពី {title}។" if article.get("language") == "km" else f"Tell me about {title}."
    response = text[:1500]  # Limit length

    return format_instruction(instruction, response)


def build_dataset():
    all_samples = []

    # Load instruction pairs
    pairs_path = os.path.join(RAW_DIR, "instruction_pairs.jsonl")
    if os.path.exists(pairs_path):
        pairs = load_jsonl(pairs_path)
        for p in pairs:
            sample = format_instruction(p["instruction"], p["response"])
            all_samples.append(sample)
        print(f"Loaded {len(pairs)} instruction pairs")

    # Load Wikipedia
    wiki_path = os.path.join(RAW_DIR, "wikipedia_km.jsonl")
    if os.path.exists(wiki_path):
        articles = load_jsonl(wiki_path)
        for a in tqdm(articles, desc="Processing Wikipedia"):
            sample = format_wikipedia(a)
            if sample:
                all_samples.append(sample)
        print(f"Loaded {len(articles)} Wikipedia articles")

    # Shuffle
    random.shuffle(all_samples)

    # Split train/val
    split = int(len(all_samples) * 0.95)
    train = all_samples[:split]
    val = all_samples[split:]

    # Save
    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path = os.path.join(OUT_DIR, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} samples → {train_path}")
    print(f"Val:   {len(val)} samples → {val_path}")
    return train, val


if __name__ == "__main__":
    print("=== Angkor LLM Dataset Preparation ===")
    build_dataset()
    print("Done!")

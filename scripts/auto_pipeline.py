"""
Angkor LLM — Automatic Data Collection & Training Pipeline
Run this script to automatically collect new Khmer data and push to Hugging Face
"""

import requests
import json
import os
import time
import random
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import HfApi, login
from xml.etree import ElementTree as ET

# ================================
# CONFIG
# ================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = "tiloukim/angkor-llm-dataset"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
WIKIPEDIA_LIMIT = 500  # articles per run
CC100_LIMIT = 5000     # sentences from CC-100 Khmer
MC4_LIMIT = 5000       # sentences from mC4 Khmer

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are AngkorAI (Angkor LLM), Cambodia's first bilingual AI assistant. You speak both Khmer and English fluently. When users write in Khmer, respond in Khmer. When they write in English, respond in English."

# ================================
# STEP 1: COLLECT WIKIPEDIA
# ================================
def collect_wikipedia():
    print("\n[1/6] Collecting Khmer Wikipedia articles...")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "AngkorLLM/1.0 (https://angkorai.ai) python-requests"
    })

    url = "https://km.wikipedia.org/w/api.php"
    articles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": 50,
        "apnamespace": 0,
        "apfilterredir": "nonredirects",
    }
    apcontinue = None
    count = 0

    while count < WIKIPEDIA_LIMIT:
        if apcontinue:
            params["apcontinue"] = apcontinue
        res = session.get(url, params=params)
        data = res.json()
        pages = data["query"]["allpages"]

        for page in tqdm(pages, desc=f"  Articles: {count}"):
            title = page["title"]
            content_res = session.get(url, params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            })
            content_data = content_res.json()
            for _, p in content_data["query"]["pages"].items():
                text = p.get("extract", "").strip()
                if text and len(text) > 100:
                    articles.append({
                        "source": "wikipedia_km",
                        "title": title,
                        "text": text,
                    })
            count += 1
            time.sleep(0.05)

        if "continue" in data:
            apcontinue = data["continue"]["apcontinue"]
        else:
            break

    path = os.path.join(RAW_DIR, "wikipedia_km.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"  Saved {len(articles)} Wikipedia articles")
    return articles


# ================================
# STEP 2: COLLECT KHMER NEWS (Multiple Sources)
# ================================
def collect_news():
    print("\n[2/6] Collecting Khmer news from multiple sources...")
    news = []

    rss_sources = [
        {"url": "https://www.rfa.org/khmer/rss2.xml",           "name": "rfa_khmer"},
        {"url": "https://www.voacambodia.com/api/zmpqeipqki",   "name": "voa_khmer"},
        {"url": "https://www.khmertimeskh.com/feed/",           "name": "khmer_times"},
        {"url": "https://freshnewsasia.com/feed/",              "name": "fresh_news"},
    ]

    for source in rss_sources:
        try:
            res = requests.get(source["url"], timeout=10, headers={
                "User-Agent": "AngkorLLM/1.0 (https://angkorai.ai)"
            })
            root = ET.fromstring(res.content)
            count = 0
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                desc = item.findtext("description", "").strip()
                if title and desc and len(desc) > 50:
                    news.append({
                        "source": source["name"],
                        "title": title,
                        "text": f"{title}. {desc}",
                    })
                    count += 1
            print(f"  {source['name']}: {count} articles")
        except Exception as e:
            print(f"  Warning: Could not fetch {source['name']}: {e}")

    path = os.path.join(RAW_DIR, "news_km.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for n in news:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    print(f"  Saved {len(news)} total news articles")
    return news


# ================================
# STEP 3: COLLECT CC-100 KHMER (Hugging Face)
# ================================
def collect_cc100():
    print("\n[3/6] Collecting CC-100 Khmer dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("cc100", lang="km", split="train", streaming=True, trust_remote_code=True)
        texts = []
        for i, sample in enumerate(dataset):
            if i >= CC100_LIMIT:
                break
            text = sample.get("text", "").strip()
            if text and len(text) > 100:
                texts.append({
                    "source": "cc100_km",
                    "text": text,
                })

        path = os.path.join(RAW_DIR, "cc100_km.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        print(f"  Saved {len(texts)} CC-100 Khmer samples")
        return texts
    except Exception as e:
        print(f"  Warning: Could not load CC-100: {e}")
        return []


# ================================
# STEP 4: COLLECT mC4 KHMER (Hugging Face)
# ================================
def collect_mc4():
    print("\n[4/6] Collecting mC4 Khmer dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("mc4", "km", split="train", streaming=True, trust_remote_code=True)
        texts = []
        for i, sample in enumerate(dataset):
            if i >= MC4_LIMIT:
                break
            text = sample.get("text", "").strip()
            if text and len(text) > 100:
                texts.append({
                    "source": "mc4_km",
                    "text": text,
                })

        path = os.path.join(RAW_DIR, "mc4_km.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        print(f"  Saved {len(texts)} mC4 Khmer samples")
        return texts
    except Exception as e:
        print(f"  Warning: Could not load mC4: {e}")
        return []


# ================================
# STEP 5: FORMAT DATASET
# ================================
def format_dataset(articles, news, cc100_texts, mc4_texts):
    print("\n[5/6] Formatting dataset...")
    samples = []

    # Format Wikipedia as Q&A
    for a in articles:
        title = a["title"]
        text = a["text"][:1500]
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"ប្រាប់ខ្ញុំអំពី {title}។"},
                {"role": "assistant", "content": text},
            ]
        })

    # Format news as summaries
    for n in news:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"សង្ខេបព័ត៌មាននេះ: {n['title']}"},
                {"role": "assistant", "content": n["text"]},
            ]
        })

    # Format CC-100 as reading comprehension
    for t in cc100_texts:
        text = t["text"][:1500]
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "អានអត្ថបទខ្មែរនេះ ហើយបកស្រាយវាឱ្យខ្ញុំ។"},
                {"role": "assistant", "content": text},
            ]
        })

    # Format mC4 as reading comprehension
    for t in mc4_texts:
        text = t["text"][:1500]
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "ជួយសង្ខេបអត្ថបទនេះជាភាសាខ្មែរ។"},
                {"role": "assistant", "content": text},
            ]
        })

    # Load and repeat founder + greeting instruction pairs 50x to ensure they dominate
    instruction_path = os.path.join(RAW_DIR, "instruction_pairs.jsonl")
    founder_samples = []
    if os.path.exists(instruction_path):
        with open(instruction_path, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                founder_samples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": pair["instruction"]},
                        {"role": "assistant", "content": pair["response"]},
                    ]
                })
        # Repeat 50 times so founder answers dominate training
        repeated = founder_samples * 50
        samples.extend(repeated)
        print(f"  Founder/greeting pairs: {len(founder_samples)} x 50 = {len(repeated)} samples")

    # Shuffle
    random.shuffle(samples)

    # Split train/val
    split = int(len(samples) * 0.95)
    train = samples[:split]
    val = samples[split:]

    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    val_path = os.path.join(PROCESSED_DIR, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Wikipedia samples:  {len(articles)}")
    print(f"  News samples:       {len(news)}")
    print(f"  CC-100 samples:     {len(cc100_texts)}")
    print(f"  mC4 samples:        {len(mc4_texts)}")
    print(f"  Total Train:        {len(train)}")
    print(f"  Total Val:          {len(val)}")
    return train, val


# ================================
# STEP 6: PUSH TO HUGGING FACE
# ================================
def push_to_huggingface():
    print("\n[6/6] Pushing to Hugging Face...")
    if not HF_TOKEN:
        print("  ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token")
        return

    login(token=HF_TOKEN)
    api = HfApi()

    api.upload_file(
        path_or_fileobj=os.path.join(PROCESSED_DIR, "train.jsonl"),
        path_in_repo="train.jsonl",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=os.path.join(PROCESSED_DIR, "val.jsonl"),
        path_in_repo="val.jsonl",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )

    print(f"  Dataset pushed to: https://huggingface.co/datasets/{HF_DATASET_REPO}")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Angkor LLM — Auto Data Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    articles = collect_wikipedia()
    news = collect_news()
    cc100_texts = collect_cc100()
    mc4_texts = collect_mc4()
    format_dataset(articles, news, cc100_texts, mc4_texts)
    push_to_huggingface()

    print("\n✓ Pipeline complete! New data is ready on Hugging Face.")
    print("  Open Colab and run the training notebook to retrain Angkor LLM.")

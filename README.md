# 🏛️ Angkor LLM

Cambodia's first fine-tuned bilingual language model — built on top of Sailor2 (SEA LLM).

**Creator:** Mr. Tilou Kim (លោក ទីលូ គីម) · Khmer-American

---

## Project Structure

```
angkor-llm/
├── scripts/
│   ├── collect_data.py      # Collect Khmer data from Wikipedia & news
│   └── prepare_dataset.py   # Format data for fine-tuning
├── train/
│   └── finetune.ipynb       # Google Colab fine-tuning notebook
├── serve/
│   └── serve.py             # Local inference server
├── data/
│   ├── raw/                 # Raw collected data
│   └── processed/           # Formatted training data
└── requirements.txt
```

---

## Quick Start

### Step 1: Collect Data
```bash
pip install -r requirements.txt
python scripts/collect_data.py
python scripts/prepare_dataset.py
```

### Step 2: Fine-tune on Google Colab
1. Open `train/finetune.ipynb` in Google Colab
2. Select **T4 GPU** (free tier)
3. Run all cells
4. Push model to Hugging Face

### Step 3: Run Locally
```bash
python serve/serve.py
```

---

## Base Model

| Model | Parameters | Language Support |
|-------|-----------|-----------------|
| [Sailor2-8B-Chat](https://huggingface.co/sail/Sailor2-8B-Chat) | 8B | Khmer, Thai, Vietnamese, Indonesian, English |

---

## Fine-tuning Method

- **QLoRA** (4-bit quantization + LoRA adapters)
- **Unsloth** for 2x faster training
- Works on free Google Colab T4 GPU

---

## Roadmap

- [x] Project setup
- [ ] Khmer Wikipedia dataset collection
- [ ] Instruction pair dataset (Khmer Q&A)
- [ ] First fine-tune run
- [ ] Evaluation on Khmer benchmarks
- [ ] Push to Hugging Face Hub
- [ ] Integrate into AngkorAI app

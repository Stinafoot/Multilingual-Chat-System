# Multilingual Chat System (Qwen2.5 + NLLB)

A multilingual NLP pipeline that accepts an English question, generates an English answer with Qwen2.5, and translates it into French and Spanish using NLLB-200.

![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow) ![Python](https://img.shields.io/badge/Python-3.8+-blue)

---

## Overview

The system chains two pretrained Transformer models into a single `multilingual_chat()` function:

1. **Qwen2.5-0.5B-Instruct** generates a high-quality English answer to any NLP question
2. **NLLB-200-Distilled-600M** translates that answer into French and Spanish

No fine-tuning — both models are used off-the-shelf via Hugging Face Transformers.

---

## Models

| Model | HuggingFace ID | Type | Purpose |
|-------|---------------|------|---------|
| Qwen2.5 | `Qwen/Qwen2.5-0.5B-Instruct` | Causal LM | English answer generation |
| NLLB-200 | `facebook/nllb-200-distilled-600M` | Seq2Seq | Translation to French & Spanish |

> **Note:** The assignment spec lists `google/mt5-small` for translation. This implementation uses `facebook/nllb-200-distilled-600M` instead, which is purpose-built for translation across 200 languages and provides higher-quality output.

---

## Pipeline

```
User question (English)
        │
  Qwen2.5-0.5B-Instruct
  (apply_chat_template → generate)
        │
  English answer
        ├──→ NLLB (eng_Latn → fra_Latn) → French answer
        └──→ NLLB (eng_Latn → spa_Latn) → Spanish answer
        │
  Print all three
```

---

## Example Output

```
Your question in English: "Explain what a neural network is"

Answer in English: A neural network is an artificial intelligence model that mimics
the structure and function of biological neurons in the human brain...

Answer in French: Un réseau neural est un modèle d'intelligence artificielle qui imite
la structure et la fonction des neurones biologiques dans le cerveau humain...

Answer in Spanish: Una red neuronal es un modelo de inteligencia artificial que imita
la estructura y función de las neuronas biológicas en el cerebro humano...
```

---

## Key Functions

```python
run_huggingface_qwen()          # Loads Qwen tokenizer + model
run_huggingface_nllb()          # Loads NLLB tokenizer + model
translate_nllb(text, lang)      # "French" or "Spanish"
multilingual_chat(text)         # End-to-end pipeline
```

### Translation Language Codes

| Language | NLLB Code |
|----------|-----------|
| English (source) | `eng_Latn` |
| French | `fra_Latn` |
| Spanish | `spa_Latn` |

---

## Test Questions Used

- `"Explain what a neural network is"`
- `"Please explain vanishing gradient and exploding gradient"`
- `"What is the difference between an RNN and an LSTM?"`
- `"Why are Transformers better at handling long-range dependencies?"`

---

## Requirements

```bash
pip install transformers torch accelerate
```

**Memory:** ~2 GB for Qwen2.5-0.5B + ~2.5 GB for NLLB-600M. GPU optional but speeds up generation significantly.

## Usage

```bash
jupyter notebook Part2.ipynb
```

---

## Files

| File | Description |
|------|-------------|
| `Part2.ipynb` | Full notebook with model loading, translation functions, and test outputs |

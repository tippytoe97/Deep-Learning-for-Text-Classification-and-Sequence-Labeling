# Deep Learning for Text Classification and Sequence Labeling

This project explores multiple deep learning models for natural language processing tasks including:
- Text classification (e.g. clickbait detection, scientific article categorization)
- Sequence labeling (e.g. Part-of-Speech tagging, Named Entity Recognition)

The project was built in **PyTorch** using models like **LSTM**, **LSTM with Attention**, **BERT**, and **DistilBERT**.

---

## What This Project Covers

### Text Classification
- Trained **LSTM** and **LSTM with Attention** models on:
  - Clickbait headline dataset (binary classification)
  - Web of Science article dataset (multi-class classification)

- Fine-tuned **BERT** models on the same datasets for comparison.

### Sequence Labeling
- Implemented BERT-based models for:
  - **Part-of-Speech (POS) tagging**
  - **Named Entity Recognition (NER)**

- Used the **CoNLL-2003** dataset to evaluate performance.

### Knowledge Distillation
- Compared full-size **BERT** with **DistilBERT** to evaluate model size, efficiency, and accuracy tradeoffs.

---

## Project Structure
-  lstm.py # LSTM and LSTM + Attention models
-  bert.py # BERT classifier for text classification
-  sequence_labeling.py # Sequence labeling model (POS, NER)
-  distilbert.py # DistilBERT implementation and comparison
-  data/ # Preprocessed datasets
-  requirements.txt # Python dependencies
-  README.md # Project documentation

  
---

## Results Snapshot

| Task                      | Model              | Accuracy / F1 |
|---------------------------|--------------------|----------------|
| Clickbait Classification | LSTM + Attention   | ~96% Accuracy  |
| Article Classification   | BERT               | ~94% Accuracy  |
| POS Tagging              | BERT               | High F1 Score  |
| NER                      | DistilBERT         | Slightly lower F1 than BERT |

(*Note: Metrics may vary depending on model config and random seed.*)

---

## Tools & Libraries
- Python, PyTorch, Transformers (HuggingFace)
- BERT, DistilBERT
- Attention mechanisms
- Sequence labeling techniques

---

## 🧑‍💻 What I Learned
- How to implement attention-based RNNs
- How transformer-based models improve NLP tasks
- The tradeoffs between model size, speed, and accuracy (BERT vs. DistilBERT)
- How to prepare data for both classification and sequence labeling

---

*This project was originally built as part of a deep learning course at Georgia Tech.*


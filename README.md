# Agentic Reasoning System

## Overview
An ML-based system for solving reasoning and math MCQs using TF-IDF, classification, and symbolic tools.

## Features
- Problem decomposition
- TF-IDF feature extraction  
- Pairwise ranking approach
- Reasoning trace generation
- Tool integration for arithmetic

## Performance
- Accuracy: [~0.2468]
- Macro F1: [~0.1746]

## Architecture
[# 2.1 Overall Pipeline  
1. **Preprocessing & Pairwise Construction**  
   - From `train.csv`, we transform each question into multiple (question, option) pairs with binary labels (1 = correct option, 0 = incorrect).  
2. **Feature Engineering**  
   - TF-IDF on concatenated (question + option) texts  
   - Additional features: token overlap ratio, number count features, operator presence  
   - Semantic similarity (optional): sentence embeddings and cosine similarity  
3. **Model Training**  
   - Base classifier (Logistic Regression or LightGBM) trained to score each (question, option) pair  
4. **Tool & Reasoning Modules**  
   - Symbolic solver / calculator module using `sympy` / Python math when arithmetic or expressions are detected  
   - Verification: cross-check outputs or validate computed results  
5. **Inference & Aggregation**  
   - For each test question, compute scores for all options, incorporate tool outputs if available  
   - Select the option with highest score  
6. **Reasoning Trace Generation**  
   - For each prediction, log a small trace: feature contributions, tool results, decision reasoning  ]

##  Problem Statement
Given:
- `train.csv` → Contains questions, options, and correct answers.
- `test.csv` → Contains questions and options without correct answers.

Goal:
- Predict the correct option for each question in `test.csv`.
- Save predictions in `output.csv` in the format:

---

##  Folder Structure

Math-Solver/
├─ data/
│   ├─ train.csv
│   ├─ test.csv
│   └─ output.csv (generated after prediction)
├─ src/
│   ├─ train_model.py   # Trains the model
│   ├─ predict.py       # Generates predictions
│   └─ utils.py         # Optional helper functions
├─ requirements.txt      # Dependencies
├─ README.md             # Project documentation
└─ model.pkl             # Saved trained model

---
# Product_Dataset

This folder contains test datasets used for product behavior analysis.

## Contents

-#`product_Dataset/`
 campus_card_swipes.csv
face_embeddings.csv
wifi_logs.csv
cctv_frames.csv
free text notes.csv
lab bookings.csv
library checkouts.csv
  - 


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sahelsasmal705/Agentic-Reasoning-System.git
   cd Math-solver
 
   ---

**README.md**, save, then run:
```powershell
git add README.md
git commit -m "Updated README"
git push origin main
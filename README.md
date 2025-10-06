# Hackathon Round 1 Submission

##  Overview
This project solves reasoning and math-based multiple-choice questions by predicting the correct option using a machine learning pipeline.  
It uses **TF-IDF for feature extraction** and **Logistic Regression for classification**.

---

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

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sahelsasmal705/Math-solver.git
   cd Math-solver
 
   ---

**README.md**, save, then run:
```powershell
git add README.md
git commit -m "Updated README"
git push origin main
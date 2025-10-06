"""
utils.py
Helper functions for Agentic Reasoning System
Author: Sahel
"""

import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------
# 1. Load CSV safely
# ---------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV data with error handling."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {path} successfully. Shape = {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return pd.DataFrame()

# ---------------------------------------------------
# 2. Combine text fields into one column
# ---------------------------------------------------
def make_text(row) -> str:
    """Combine problem statement and options into one text string."""
    opts = " ".join([f"{i+1}) {row.get(f'answer_option_{i+1}', '')}" for i in range(5)])
    return f"{row.get('topic','')}. Problem: {row.get('problem_statement','')} Options: {opts}"

# ---------------------------------------------------
# 3. Preprocess DataFrame
# ---------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe by adding 'text' column."""
    df = df.copy()
    df['text'] = df.apply(make_text, axis=1)
    df['text'] = df['text'].fillna("")
    return df

# ---------------------------------------------------
# 4. Model save/load utilities
# ---------------------------------------------------
def save_model(model, path: str):
    """Save model using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved to {path}")

def load_model(path: str):
    """Load model using joblib."""
    if Path(path).exists():
        model = joblib.load(path)
        print(f"✅ Loaded model from {path}")
        return model
    else:
        print(f"❌ Model not found: {path}")
        return None

# ---------------------------------------------------
# 5. Reasoning trace generator (optional)
# ---------------------------------------------------
def generate_trace(pred_probs, classes):
    """
    Convert model probabilities into readable reasoning trace.
    Example: 'Top choices → 3 (0.61), 1 (0.23), 5 (0.16)'
    """
    if pred_probs is None:
        return "No trace available"
    sorted_idx = pred_probs.argsort()[::-1][:3]
    trace = ", ".join([f"{classes[i]} ({pred_probs[i]:.2f})" for i in sorted_idx])
    return f"Top choices → {trace}"

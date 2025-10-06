import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]         # project root
MODEL_PATH = ROOT / "model.pkl"                    # adjust if different
TEST_PATH = ROOT / "data" / "test.csv"
OUT_PATH = ROOT / "data" / "output.csv"

print("📦 Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)
print("✅ Model loaded.")

print("📄 Reading test CSV:", TEST_PATH)
test_df = pd.read_csv(TEST_PATH, encoding="utf-8", na_filter=False)
print(f"✅ Test rows: {len(test_df)}")

# Robustly combine text: handle missing/NaN and dynamic option columns
def combine_text(row):
    # Find all columns that start with 'answer_option_'
    option_cols = [c for c in row.index if c.startswith("answer_option_")]
    options = []
    for c in option_cols:
        val = row.get(c, "")
        options.append(str(val) if pd.notnull(val) else "")
    ps = str(row.get("problem_statement", ""))
    return ps + " " + " ".join(options)

test_df["text"] = test_df.apply(combine_text, axis=1)

print("🤖 Predicting...")
preds = model.predict(test_df["text"])

# Build required output format
test_df["solution"] = [f"Option {p}" for p in preds]
# NOTE: The required header in your sample was: 'correct option' (with a space)
test_df["correct option"] = preds

out_cols = ["topic", "problem_statement", "solution", "correct option"]
missing = [c for c in out_cols if c not in test_df.columns]
if missing:
    print("⚠️ Missing columns in test.csv:", missing)
    # Fill with empty columns if needed
    for c in missing:
        test_df[c] = ""

test_df[out_cols].to_csv(OUT_PATH, index=False, encoding="utf-8")
print("💾 Predictions saved to:", OUT_PATH)
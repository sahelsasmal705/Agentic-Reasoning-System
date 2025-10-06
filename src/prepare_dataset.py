# src/prepare_dataset.py
import pandas as pd
from pathlib import Path
from utils import combine_text_row
from sklearn.model_selection import train_test_split
import json

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "train.csv"
TEST_CSV  = ROOT / "data" / "test.csv"
OUT_DIR   = ROOT / "data"

def row_to_prompt(row):
    # Build a readable prompt with numbered options
    option_cols = [c for c in row.index if c.startswith("answer_option_")]
    opts = []
    for i, c in enumerate(option_cols, 1):
        val = str(row.get(c, ""))
        opts.append(f"{i}) {val}")
    options_str = "\n".join(opts)
    ps = str(row.get("problem_statement", ""))
    return (
        "You are a careful math/reasoning solver.\n"
        "Choose the correct option number from the given choices.\n\n"
        f"Question:\n{ps}\n\nOptions:\n{options_str}\n\n"
        "Respond strictly as: Answer: <number>\n"
        "For example: Answer: 3\n"
    )

def build_sft_rows(df, has_label: bool):
    rows = []
    for _, r in df.iterrows():
        prompt = row_to_prompt(r)
        if has_label:
            label_col = "correct_option_number" if "correct_option_number" in df.columns else "correct option"
            ans_num = int(r[label_col])
            completion = f"Answer: {ans_num}"
        else:
            completion = ""  # unknown at test time
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8", na_filter=False)
    # Optional: keep only rows with valid labels
    label_col = "correct_option_number" if "correct_option_number" in train_df.columns else "correct option"
    train_df = train_df[train_df[label_col].notnull()].copy()
    train_df[label_col] = train_df[label_col].astype(int)

    # Split train/val
    tr_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df[label_col])

    tr_rows  = build_sft_rows(tr_df, has_label=True)
    val_rows = build_sft_rows(val_df, has_label=True)

    write_jsonl(OUT_DIR / "sft_train.jsonl", tr_rows)
    write_jsonl(OUT_DIR / "sft_val.jsonl",   val_rows)

    # Prepare test prompts only
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8", na_filter=False)
    test_rows = build_sft_rows(test_df, has_label=False)
    write_jsonl(OUT_DIR / "sft_test.jsonl",  test_rows)

    print("✅ Wrote:", OUT_DIR / "sft_train.jsonl", OUT_DIR / "sft_val.jsonl", OUT_DIR / "sft_test.jsonl")

if __name__ == "__main__":
    main()
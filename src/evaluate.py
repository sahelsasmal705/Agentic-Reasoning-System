# src/evaluate.py
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

from utils import combine_text_row

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "output.csv"
MODEL_PATH = ROOT / "model.pkl"

def get_label_col(df: pd.DataFrame) -> str:
    if "correct_option_number" in df.columns:
        return "correct_option_number"
    if "correct option" in df.columns:
        return "correct option"
    raise KeyError("Expected 'correct_option_number' or 'correct option' in labels file.")

def eval_on_train_split():
    print(f"📄 Loading train from: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8", na_filter=False)
    df["text"] = df.apply(combine_text_row, axis=1)

    label_col = get_label_col(df)
    X = df["text"]
    y = pd.to_numeric(df[label_col], errors="coerce").astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("🧠 Training quick model for evaluation...")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Macro F1: {macro_f1:.4f}")
    print("\n— Classification Report —")
    print(classification_report(y_val, y_pred, digits=4))
    print("— Confusion Matrix —")
    print(confusion_matrix(y_val, y_pred))

def eval_output_vs_labels(labels_path: Path):
    """
    Compares your predictions in data/output.csv with a labels CSV.
    The labels CSV must include either 'correct_option_number' or 'correct option'.
    We align rows by order (assuming same ordering as test.csv used for output).
    """
    print(f"📄 Loading predictions from: {OUTPUT_PATH}")
    pred_df = pd.read_csv(OUTPUT_PATH, encoding="utf-8", na_filter=False)

    print(f"📄 Loading labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path, encoding="utf-8", na_filter=False)

    label_col = get_label_col(labels_df)

    # Align by row order (simplest). If you have a stable key, you can merge on it.
    if len(pred_df) != len(labels_df):
        print(f"⚠️ Row count mismatch: predictions={len(pred_df)} vs labels={len(labels_df)}")
        print("   Ensure both files are aligned to the same test set ordering.")
        return

    # Predicted column can be 'correct option' (per your output format)
    pred_col = "correct option" if "correct option" in pred_df.columns else None
    if pred_col is None:
        raise KeyError("Expected 'correct option' in output.csv")

    y_true = pd.to_numeric(labels_df[label_col], errors="coerce").astype(int)
    y_pred = pd.to_numeric(pred_df[pred_col], errors="coerce").astype(int)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test Macro F1: {macro_f1:.4f}")
    print("\n— Classification Report —")
    print(classification_report(y_true, y_pred, digits=4))
    print("— Confusion Matrix —")
    print(confusion_matrix(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Macro F1")
    parser.add_argument("--on-train-split", action="store_true",
                        help="Compute Macro F1 on a validation split from train.csv")
    parser.add_argument("--on-output", action="store_true",
                        help="Compare data/output.csv with a labels CSV")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels CSV (required for --on-output)")
    args = parser.parse_args()

    if args.on_train_split:
        eval_on_train_split()
    elif args.on_output:
        if not args.labels:
            raise ValueError("Please pass --labels path/to/test_labels.csv")
        eval_output_vs_labels(Path(args.labels))
    else:
        print("Please choose one mode:\n"
              "  python src/evaluate.py --on-train-split\n"
              "  python src/evaluate.py --on-output --labels data/test_labels.csv")

if __name__ == "__main__":
    main()
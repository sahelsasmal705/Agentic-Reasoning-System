# src/evaluate.py
import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional, not used here but kept as a constant if you want to save/load models later
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "output.csv"
MODEL_PATH = ROOT / "model.pkl"

# Import user utility to combine text fields (expects utils.py in project)
from utils import combine_text_row


def get_label_col(df: pd.DataFrame) -> str:
    """
    Return the column name used for labels in df.
    Accepts either 'correct_option_number' or 'correct option'.
    Raises KeyError if neither found.
    """
    if "correct_option_number" in df.columns:
        return "correct_option_number"
    if "correct option" in df.columns:
        return "correct option"
    raise KeyError("Expected 'correct_option_number' or 'correct option' in labels file.")


def safe_to_int_series(s: pd.Series, series_name: str):
    """
    Convert a series to integer dtype for metrics. Coerce non-numeric to NaN and drop them.
    Returns cleaned series (index-preserving) and number of dropped rows.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    before = len(numeric)
    numeric = numeric.dropna().astype(int)
    after = len(numeric)
    dropped = before - after
    if dropped:
        print(f"⚠️ Dropped {dropped} non-numeric / missing rows from '{series_name}'.")
    return numeric


def eval_on_train_split(train_path: Path = TRAIN_PATH):
    if not train_path.exists():
        print(f"❌ Train file not found: {train_path}")
        return

    print(f"📄 Loading train from: {train_path}")
    df = pd.read_csv(train_path, encoding="utf-8", na_filter=False)

    # Combine row text (expected to be implemented in utils.combine_text_row)
    df["text"] = df.apply(combine_text_row, axis=1)

    label_col = get_label_col(df)
    X = df["text"]
    y_raw = df[label_col]

    # Convert labels to integers; drop rows where conversion failed
    y = pd.to_numeric(y_raw, errors="coerce")
    mask = y.notna()
    if mask.sum() < len(y):
        print(f"⚠️ Found {len(y) - mask.sum()} rows with missing/non-numeric labels; these will be dropped.")
    X = X[mask]
    y = y[mask].astype(int)

    # If too few classes or samples, warn and stop
    if len(X) < 2:
        print("❌ Not enough data after cleaning to perform train/validation split.")
        return

    print("🔀 Splitting train/validation (80/20) with stratification...")
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
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ Macro F1: {macro_f1:.4f}")
    print("\n— Classification Report —")
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))
    print("— Confusion Matrix —")
    print(confusion_matrix(y_val, y_pred))


def eval_output_vs_labels(labels_path: Path, output_path: Path = OUTPUT_PATH):
    """
    Compares predictions in output_path with a labels CSV.
    The labels CSV must include either 'correct_option_number' or 'correct option'.
    Rows are aligned by order (assumes same ordering as the test set used to produce output.csv).
    """
    if not output_path.exists():
        print(f"❌ Predictions file not found: {output_path}")
        return
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        return

    print(f"📄 Loading predictions from: {output_path}")
    pred_df = pd.read_csv(output_path, encoding="utf-8", na_filter=False)

    print(f"📄 Loading labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path, encoding="utf-8", na_filter=False)

    label_col = get_label_col(labels_df)

    if len(pred_df) != len(labels_df):
        print(f"⚠️ Row count mismatch: predictions={len(pred_df)} vs labels={len(labels_df)}")
        print("   Ensure both files are aligned to the same test set ordering.")
        # continue but only compare up to min length
        n = min(len(pred_df), len(labels_df))
        pred_df = pred_df.iloc[:n].reset_index(drop=True)
        labels_df = labels_df.iloc[:n].reset_index(drop=True)

    # Predicted column can be 'correct option' (per your output format)
    if "correct option" not in pred_df.columns:
        raise KeyError("Expected column 'correct option' in output.csv with your predictions.")

    y_true_raw = labels_df[label_col]
    y_pred_raw = pred_df["correct option"]

    # Convert to numeric and drop rows with missing labels/preds
    y_true = pd.to_numeric(y_true_raw, errors="coerce")
    y_pred = pd.to_numeric(y_pred_raw, errors="coerce")

    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < len(y_true):
        print(f"⚠️ Dropping {len(y_true) - mask.sum()} rows with missing/non-numeric labels or predictions.")
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    if len(y_true) == 0:
        print("❌ No valid rows to evaluate after cleaning labels/predictions.")
        return

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test Macro F1: {macro_f1:.4f}")
    print("\n— Classification Report —")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("— Confusion Matrix —")
    print(confusion_matrix(y_true, y_pred))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluate predictions and compute Macro F1 / Accuracy")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--on-train-split", action="store_true",
                       help="Compute Macro F1 on a validation split from train.csv")
    group.add_argument("--on-output", action="store_true",
                       help="Compare data/output.csv with a labels CSV (requires --labels)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels CSV (required for --on-output)")
    return parser.parse_args(argv)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    try:
        if args.on_train_split:
            eval_on_train_split()
        elif args.on_output:
            if not args.labels:
                raise ValueError("Please pass --labels path/to/test_labels.csv when using --on-output")
            eval_output_vs_labels(Path(args.labels))
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"❌ Empty or invalid CSV file: {e}")
    except KeyError as e:
        print(f"❌ Column error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

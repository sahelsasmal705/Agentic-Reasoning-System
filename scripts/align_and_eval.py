# scripts/align_and_eval.py
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(".")
pred_path = ROOT / "data" / "output.csv"
labels_path = ROOT / "data" / "train.csv"   # change if you meant a different file

pred = pd.read_csv(pred_path, encoding="utf-8", na_filter=False)
labels = pd.read_csv(labels_path, encoding="utf-8", na_filter=False)

print("Pred rows:", len(pred), "Labels rows:", len(labels))
print("Pred columns:", list(pred.columns))
print("Label columns:", list(labels.columns))

# common id candidates
cands = ["id","ID","Id","question_id","questionId","qid","test_id","index"]
common = [c for c in cands if c in pred.columns and c in labels.columns]
if not common:
    print("\nNo obvious common ID columns found among", cands)
    print("Cannot align by id. Consider re-generating output.csv matching the test file ordering.")
else:
    key = common[0]
    print("\nAligning on key:", key)
    # drop duplicates on key to be safe
    pred = pred.drop_duplicates(subset=[key]).set_index(key)
    labels = labels.drop_duplicates(subset=[key]).set_index(key)
    # intersect keys
    idx = pred.index.intersection(labels.index)
    print("Common keys count:", len(idx))
    if len(idx) == 0:
        print("No overlapping keys found between predictions and labels.")
    else:
        y_true = pd.to_numeric(labels.loc[idx, "correct_option_number"] if "correct_option_number" in labels.columns else labels.loc[idx,"correct option"], errors="coerce").astype(int)
        pred_col = "correct option" if "correct option" in pred.columns else ( "prediction" if "prediction" in pred.columns else None )
        if not pred_col:
            # try to find numeric column (excluding id)
            nonid = [c for c in pred.columns if c != key]
            print("Trying to use first non-id column as predictions:", nonid)
            pred_col = nonid[0]
        y_pred = pd.to_numeric(pred.loc[idx, pred_col], errors="coerce").astype(int)
        print("\nEvaluation on matched subset:")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))
        print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))
        print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))

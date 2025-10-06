# scripts/infer.py
import os
import pandas as pd
import joblib
from rule_solver import solve_sdt, solve_algebra, extract_numbers  # local file
import numpy as np

os.makedirs("models", exist_ok=True)
vec = joblib.load("models/vectorizer.joblib")
clf = joblib.load("models/model.joblib")

# read test
test_path = "data/test.csv"
if not os.path.exists(test_path):
    raise SystemExit("Put test.csv into data/test.csv first")

test = pd.read_csv(test_path)

def make_text(row):
    opts = " ".join([f"{i+1}) {row.get(f'answer_option_{i+1}','')}" for i in range(5)])
    return f"{row.get('topic','')}. Problem: {row['problem_statement']} Options: {opts}"

test['text'] = test.apply(make_text, axis=1)

# Try to detect sample output format
sample_output = None
sample_path = "data/sample_output.csv"
if os.path.exists(sample_path):
    sample_output = pd.read_csv(sample_path)
    print("Found sample_output.csv — will respect its column names where possible.")

preds = []
explanations = []

from scripts.rule_solver import solve_sdt as rule_sdt, solve_algebra as rule_alg

for idx, row in test.iterrows():
    opts = [row.get(f"answer_option_{i+1}", "") for i in range(5)]
    problem_text = row['problem_statement']
    # 1) try numeric SDT rule
    r = rule_sdt(problem_text, opts)
    if r.get("matched"):
        pred = r.get("option")
        expl = f"rule_numeric: computed {r.get('computed_value')} matched option {pred} (opt_val={r.get('option_value')}, diff={r.get('diff')})"
    else:
        # try algebraic
        r2 = rule_alg(problem_text, opts)
        if r2.get("matched"):
            pred = r2.get("option")
            expl = f"rule_algebra: matched option {pred} (computed {r2.get('computed_value')})"
        else:
            # fallback to ML
            x = vec.transform([row['text']])
            prob = clf.predict_proba(x)[0]
            pred = int(clf.predict(x)[0])
            top3 = np.argsort(prob)[::-1][:3] + 1
            top3_scores = [f"{(prob[i-1]):.3f}" for i in top3]
            expl = f"ml: pred={pred}; top3={(list(top3))} scores={top3_scores}"

    preds.append(pred)
    explanations.append(expl)

test['predicted_option'] = preds
test['explanation'] = explanations

# Build output file
if sample_output is not None:
    # find the likely label column in sample output
    sample_cols = sample_output.columns.tolist()
    label_col = None
    for c in sample_cols:
        if 'correct' in c.lower() or 'answer' in c.lower() or 'option' in c.lower():
            label_col = c
            break
    if label_col is None:
        # fallback
        label_col = 'correct_option_number'
    out = sample_output.copy()
    # map predictions to out rows by index if same length, else create new mapping
    if len(out) == len(test):
        out[label_col] = test['predicted_option'].values
        out['explanation'] = test['explanation'].values
    else:
        # create standard {id, label}
        out = test.reset_index().rename(columns={'index':'id'})[['id']]
        out[label_col] = test['predicted_option'].values
        out['explanation'] = test['explanation'].values
else:
    # fallback basic CSV
    if 'id' in test.columns:
        out = test[['id']].copy()
        out['correct_option_number'] = test['predicted_option']
        out['explanation'] = test['explanation']
    else:
        out = test.reset_index().rename(columns={'index':'id'})[['id']]
        out['correct_option_number'] = test['predicted_option']
        out['explanation'] = test['explanation']

out_path = "data/output.csv"
out.to_csv(out_path, index=False)
print("Saved predictions to", out_path)
print("Done.")
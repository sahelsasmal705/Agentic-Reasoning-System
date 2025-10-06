# scripts/train_baseline.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/train.csv")

def make_text(row):
    opts = " ".join([f"{i+1}) {row.get(f'answer_option_{i+1}','')}" for i in range(5)])
    return f"{row.get('topic','')}. Problem: {row['problem_statement']} Options: {opts}"

df['text'] = df.apply(make_text, axis=1)
X = df['text'].fillna("")
y = df['correct_option_number'].astype(int)

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

tf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)
Xtr_t = tf.fit_transform(Xtr)
Xval_t = tf.transform(Xval)

clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=3000, class_weight='balanced', C=1.0, random_state=42)
clf.fit(Xtr_t, ytr)

y_pred = clf.predict(Xval_t)
print(classification_report(yval, y_pred, digits=4))
print("macro-F1:", f1_score(yval, y_pred, average='macro'))

joblib.dump(tf, "models/vectorizer.joblib")
joblib.dump(clf, "models/model.joblib")
print("Saved models to models/")
print("Done.")
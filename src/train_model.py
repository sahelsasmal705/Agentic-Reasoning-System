print("Starting training...")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load training data
df = pd.read_csv("data/train.csv")
print("Data loaded:", df.shape)

# Combine question + options
def combine_text(row):
    options = []
    for i in range(1, 6):
        val = row.get(f"answer_option_{i}", "")
        options.append(str(val) if pd.notnull(val) else "")
    return str(row["problem_statement"]) + " " + " ".join(options)

df["text"] = df.apply(combine_text, axis=1)
X = df["text"]
y = df["correct_option_number"]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
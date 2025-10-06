# scripts/eda.py
import pandas as pd
import os

d = "data"
print("Files in data/:", os.listdir(d))

train = pd.read_csv(os.path.join(d, "train.csv"))
print("Train shape:", train.shape)
print("Columns:", train.columns.tolist())
print("Missing per column:\n", train.isnull().sum())
print("\nLabel counts:\n", train['correct_option_number'].value_counts())
print("\nSample rows:\n", train.head(5).T)

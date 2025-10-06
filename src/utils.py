# src/utils.py
import pandas as pd

def combine_text_row(row) -> str:
    """
    Combine problem_statement + all answer_option_* columns into one text string.
    Handles missing values and non-strings.
    """
    ps = str(row.get("problem_statement", ""))

    # Dynamically gather all option columns (answer_option_1..5 or fewer)
    option_cols = [c for c in row.index if c.startswith("answer_option_")]
    options = []
    for c in option_cols:
        val = row.get(c, "")
        options.append(str(val) if pd.notnull(val) else "")

    return ps + " " + " ".join(options)

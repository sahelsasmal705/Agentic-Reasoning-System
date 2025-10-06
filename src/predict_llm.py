# Add at top
import argparse

# (Removed duplicate main() definition to avoid function name conflict)


# src/predict_llm.py
import time
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_CSV  = DATA_DIR / "test.csv"
OUT_CSV   = DATA_DIR / "output.csv"
LORA_DIR  = ROOT / "outputs" / "lora-gptj"  # same as train_llm_lora.py

def build_prompt(row):
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

def extract_number_from_text(text: str) -> int:
    # Parse "Answer: <n>" from model output; fallback to best effort
    for token in text.split():
        if token.isdigit():
            n = int(token)
            if 1 <= n <= 5:
                return n
    return 1  # default/fallback

def main():
    df = pd.read_csv(TEST_CSV, encoding="utf-8", na_filter=False)

    tok = AutoTokenizer.from_pretrained(str(LORA_DIR))
    base = AutoModelForCausalLM.from_pretrained(str(LORA_DIR), device_map="auto")
    # If adapters are separate: base = AutoModelForCausalLM.from_pretrained(BASE_MODEL); model = PeftModel.from_pretrained(base, LORA_DIR)
    model = base

    preds = []
    soln = []
    times = []

    for _, row in df.iterrows():
        prompt = build_prompt(row)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0
        )
        gen = tok.decode(out[0], skip_special_tokens=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        # Extract last lines after prompt
        answer_text = gen[len(prompt):].strip()
        n = extract_number_from_text(answer_text)
        preds.append(n)
        soln.append(f"Option {n}")

    df["solution"] = soln
    df["correct option"] = preds

    df[["topic","problem_statement","solution","correct option"]].to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("✅ Saved predictions to:", OUT_CSV)
    print(f"⏱️ Avg inference time per question: {sum(times)/len(times):.3f}s")

if __name__ == "__main__":
    main()
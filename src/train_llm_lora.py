# src/train_llm_lora.py
import json, math
from pathlib import Path
from datasets import load_dataset # type: ignore
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # type: ignore # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_JSONL = str(DATA_DIR / "sft_train.jsonl")
VAL_JSONL   = str(DATA_DIR / "sft_val.jsonl")
OUT_DIR     = ROOT / "outputs" / "lora-gptj"

# ===== Choose your base model here =====
BASE_MODEL = "EleutherAI/gpt-j-6B"     # 6B parameters, under 13B
LOAD_IN_4BIT = True                    # set False on CPU if bitsandbytes not available
BF16 = False                           # set True if your GPU supports bf16 (A100/H100)

def get_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_raw_dataset(train_jsonl, val_jsonl):
    ds_train = load_dataset("json", data_files=train_jsonl, split="train")
    ds_val   = load_dataset("json", data_files=val_jsonl,   split="train")
    return ds_train, ds_val

def build_model(model_name):
    quant_cfg = None
    if LOAD_IN_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_cfg
    )
    return model

def format_example(ex, tok, max_len=1024):
    # Concatenate prompt + completion for causal LM training
    prompt = ex["prompt"]
    completion = ex.get("completion", "")
    text = prompt + completion
    enc = tok(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length"
    )
    # For simple SFT, we train on full text; you can mask labels for prompt if desired
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = get_tokenizer(BASE_MODEL)
    model = build_model(BASE_MODEL)

    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj","fc_in","fc_out"],  # generic; adjust per model
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds_train, ds_val = load_raw_dataset(TRAIN_JSONL, VAL_JSONL)

    def map_fn(batch):
        return format_example(batch, tok)

    ds_train = ds_train.map(map_fn, batched=False)
    ds_val   = ds_val.map(map_fn, batched=False)

    dc = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=1,              # start small; increase to 2-3 if time permits
        per_device_train_batch_size=1,   # adjust based on VRAM
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # simulate larger batch
        learning_rate=2e-4,
        fp16=LOAD_IN_4BIT,               # fp16 with 4-bit compute
        bf16=BF16,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        warmup_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tok, # type: ignore
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=dc
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    print("✅ LoRA finetuned model saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
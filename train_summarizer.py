"""
BriefSync — Fine-tuning T5-small for abstractive dialogue summarization on SAMSum.
Continues from KK's own preprocessing (text-summarizer.ipynb): random-sampled subset,
lowercase + whitespace/HTML cleaning. This script adds: tokenization, the actual
Seq2SeqTrainer fine-tuning loop, ROUGE evaluation, and saving the model in the same
"./saved_summary_model" layout the reference Summarizer-HF app.py expects to load.
"""

import re
import numpy as np
import pandas as pd
import evaluate
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

MODEL_NAME = "t5-small"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 64

# ---------------------------------------------------------------
# 1. Load + sample (same seed/sizes as KK's notebook)
# ---------------------------------------------------------------
train_data = pd.read_csv("samsum-train.csv").dropna(subset=["dialogue", "summary"])
val_data = pd.read_csv("samsum-validation.csv").dropna(subset=["dialogue", "summary"])

train_data = train_data.sample(n=800, random_state=42).reset_index(drop=True)
val_data = val_data.sample(n=100, random_state=42).reset_index(drop=True)


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip().lower()


train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"] = train_data["summary"].apply(clean_data)
val_data["dialogue"] = val_data["dialogue"].apply(clean_data)
val_data["summary"] = val_data["summary"].apply(clean_data)

# T5 needs a task prefix
train_data["input_text"] = "summarize: " + train_data["dialogue"]
val_data["input_text"] = "summarize: " + val_data["dialogue"]

train_ds = Dataset.from_pandas(train_data[["input_text", "summary"]])
val_ds = Dataset.from_pandas(val_data[["input_text", "summary"]])

print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

# ---------------------------------------------------------------
# 2. Tokenize
# ---------------------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


def preprocess(batch):
    model_inputs = tokenizer(
        batch["input_text"], max_length=MAX_INPUT_LEN, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=batch["summary"], max_length=MAX_TARGET_LEN, truncation=True, padding="max_length"
    )
    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq] for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

# ---------------------------------------------------------------
# 3. Model + Trainer
# ---------------------------------------------------------------
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
collator = DataCollatorForSeq2Seq(tokenizer, model=model)
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}


args = Seq2SeqTrainingArguments(
    output_dir="./briefsync_checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    logging_steps=20,
    report_to=[],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning...")
train_result = trainer.train()
print("Training done:", train_result.metrics)

print("Running final evaluation...")
eval_metrics = trainer.evaluate()
print("Eval metrics:", eval_metrics)

# ---------------------------------------------------------------
# 4. Save model in the layout the Summarizer-HF-style app.py expects
# ---------------------------------------------------------------
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")

with open("training_results.txt", "w") as f:
    f.write("BriefSync fine-tuning results (t5-small, SAMSum, 800 train / 100 val, 3 epochs)\n")
    f.write(f"Train metrics: {train_result.metrics}\n")
    f.write(f"Eval metrics: {eval_metrics}\n")

print("Saved model to ./saved_summary_model")

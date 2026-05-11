"""
Phase 5 + 6 — Fine-tune BERT with LoRA for resume NER.

Usage:
    pip install transformers datasets seqeval peft accelerate --break-system-packages
    python train.py

What this does:
    1. Loads your converted dataset from dataset/train.json and dataset/val.json
    2. Loads bert-base-cased with a token classification head
    3. Wraps it with LoRA adapters (Phase 6) — trains only ~0.3% of parameters
    4. Trains for 5 epochs, evaluating after each
    5. Saves the best model to models/resume-ner/
    6. Prints Precision, Recall, F1 per entity type
"""
from __future__ import annotations
import json
import os
import numpy as np
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model, TaskType
from seqeval.metrics import classification_report, f1_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-cased"
DATASET_DIR  = "dataset"
OUTPUT_DIR   = "models/resume-ner"
BATCH_SIZE   = 8       # lower if you get OOM errors
EPOCHS       = 5
LR           = 2e-4    # higher than full fine-tuning because LoRA adapters are small
MAX_LENGTH   = 512

# LoRA config — explained below
LORA_R       = 8       # rank: higher = more capacity but more parameters
LORA_ALPHA   = 32      # scaling: usually 2×r or 4×r
LORA_DROPOUT = 0.1


def load_dataset_from_json(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def compute_metrics(label_map: dict):
    """
    Returns a metrics function for the Trainer.
    Uses seqeval which is the standard library for NER evaluation.
    """
    id2label = label_map["id2label"]

    def _compute(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        true_labels  = []
        true_preds   = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_clean = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:   # special token — skip
                    continue
                true_seq.append(id2label[str(label_id)])
                pred_seq_clean.append(id2label[str(pred_id)])
            true_labels.append(true_seq)
            true_preds.append(pred_seq_clean)

        # seqeval computes entity-level F1 (not token-level)
        # This is what matters: did you extract the right entity spans?
        report = classification_report(true_labels, true_preds, output_dict=True)
        overall_f1 = f1_score(true_labels, true_preds)

        # Print per-entity breakdown
        print("\n── Entity-level results ──────────────────────────────")
        for entity, scores in report.items():
            if entity in ("micro avg", "macro avg", "weighted avg"):
                continue
            if isinstance(scores, dict):
                print(f"  {entity:15s}  "
                      f"P={scores['precision']:.3f}  "
                      f"R={scores['recall']:.3f}  "
                      f"F1={scores['f1-score']:.3f}  "
                      f"(support={int(scores['support'])})")
        print(f"\n  Overall F1: {overall_f1:.4f}")
        print("──────────────────────────────────────────────────────")

        return {"f1": overall_f1}

    return _compute


def main():
    # ── Load label map ─────────────────────────────────────────────────────────
    label_map_path = Path(DATASET_DIR) / "label_map.json"
    if not label_map_path.exists():
        print("ERROR: dataset/label_map.json not found.")
        print("Run convert_annotations.py first.")
        return

    with open(label_map_path) as f:
        label_map = json.load(f)

    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    num_labels = len(label2id)

    print(f"Labels ({num_labels}): {list(label2id.keys())}")

    # ── Load dataset ───────────────────────────────────────────────────────────
    train_path = Path(DATASET_DIR) / "train.json"
    val_path   = Path(DATASET_DIR) / "val.json"

    if not train_path.exists():
        print("ERROR: dataset/train.json not found. Run convert_annotations.py first.")
        return

    train_dataset = load_dataset_from_json(str(train_path))
    val_dataset   = load_dataset_from_json(str(val_path))

    print(f"Train: {len(train_dataset)} examples")
    print(f"Val:   {len(val_dataset)} examples")

    # ── Load tokenizer and model ───────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    # Instead of updating all 110M parameters, LoRA inserts small adapter
    # matrices into the attention layers and only trains those.
    # Result: ~300K trainable parameters instead of 110M.
    #
    # target_modules: which weight matrices to adapt.
    # For BERT, "query" and "value" in each attention head are the standard choice.
    # Adding "key" and the dense layer gives slightly better results but uses more memory.
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],   # always train the classification head fully
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: trainable params ~300K out of ~108M (0.27%)

    # ── Training arguments ─────────────────────────────────────────────────────
    # These are sensible defaults for a small dataset.
    # If your val F1 stops improving after epoch 2-3, reduce num_train_epochs.
    # If you get OOM errors, reduce per_device_train_batch_size to 4.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        warmup_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        label_names=["labels"],
    )

    # ── Data collator ──────────────────────────────────────────────────────────
    # Handles padding within each batch dynamically (more efficient than fixed padding)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, 
        data_collator=data_collator,
        compute_metrics=compute_metrics(label_map),
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    print("(This will take a few minutes on CPU, ~30 seconds per epoch on GPU)\n")
    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}/")

    # Save label map alongside model for inference
    with open(final_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\nDone. Next step: run  python test_model.py  to see predictions on new resumes.")


if __name__ == "__main__":
    main()
"""
Evaluate baseline BERT token classifier vs LoRA fine-tuned model on val.json.

Writes:
  reports/baseline_metrics.json
  reports/fine_tuned_metrics.json

Run from repository root:
  py code/scripts/evaluate_baseline_vs_lora.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from peft import PeftModel
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
MODEL_FINAL = REPO_ROOT / "models" / "resume-ner" / "final"
REPORTS_DIR = REPO_ROOT / "reports"
BASE_MODEL = "bert-base-cased"


def load_val() -> Dataset:
    path = DATASET_DIR / "val.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def load_label_map() -> dict:
    p = DATASET_DIR / "label_map.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_metrics_dict(label_map: dict, predictions: np.ndarray, labels: np.ndarray) -> dict:
    id2label = label_map["id2label"]
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_ids = np.argmax(logits, axis=2)

    # Token-level (excluding padding) — often >0 even when strict entity F1 is low
    tok_correct = 0
    tok_total = 0
    for pred_seq, label_seq in zip(pred_ids, labels):
        for pid, lid in zip(pred_seq, label_seq):
            if int(lid) == -100:
                continue
            tok_total += 1
            if int(pid) == int(lid):
                tok_correct += 1
    token_accuracy = float(tok_correct / tok_total) if tok_total else 0.0
    true_labels: list = []
    true_preds: list = []

    for pred_seq, label_seq in zip(pred_ids, labels):
        t_seq: list[str] = []
        p_seq: list[str] = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if int(label_id) == -100:
                continue
            t_seq.append(id2label[str(int(label_id))])
            p_seq.append(id2label[str(int(pred_id))])
        true_labels.append(t_seq)
        true_preds.append(p_seq)

    report = classification_report(true_labels, true_preds, output_dict=True)
    overall_f1 = float(f1_score(true_labels, true_preds))
    out: dict = {
        "overall_f1": overall_f1,
        "token_accuracy_excl_padding": token_accuracy,
        "per_entity": {},
    }
    for entity, scores in report.items():
        if entity in ("micro avg", "macro avg", "weighted avg"):
            continue
        if isinstance(scores, dict):
            out["per_entity"][entity] = {
                "precision": float(scores["precision"]),
                "recall": float(scores["recall"]),
                "f1": float(scores["f1-score"]),
                "support": int(scores["support"]),
            }
    for key in ("micro avg", "macro avg", "weighted avg"):
        if key in report and isinstance(report[key], dict):
            out[key.replace(" ", "_")] = {
                "precision": float(report[key]["precision"]),
                "recall": float(report[key]["recall"]),
                "f1": float(report[key]["f1-score"]),
            }
    return out


def evaluate_split(name: str, model, tokenizer, val_dataset: Dataset, label_map: dict) -> dict:
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=-100)
    args = TrainingArguments(
        output_dir=str(REPORTS_DIR / f"eval_tmp_{name}"),
        per_device_eval_batch_size=8,
        dataloader_drop_last=False,
        report_to="none",
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    out = trainer.predict(val_dataset)
    return build_metrics_dict(label_map, out.predictions, out.label_ids)


def main() -> int:
    if not MODEL_FINAL.is_dir():
        print(f"Fine-tuned model not found at {MODEL_FINAL}", file=sys.stderr)
        return 1

    label_map = load_label_map()
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    val_ds = load_val()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_FINAL))

    # Baseline: BERT + fresh classification head (no adapter weights)
    base = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    baseline_metrics = evaluate_split("baseline", base, tokenizer, val_ds, label_map)

    # Fine-tuned: merge LoRA into base for fast eval
    tuned_base = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    peft_model = PeftModel.from_pretrained(tuned_base, str(MODEL_FINAL))
    merged = peft_model.merge_and_unload()
    finetuned_metrics = evaluate_split("finetuned", merged, tokenizer, val_ds, label_map)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "fine_tuned_metrics.json").write_text(
        json.dumps(finetuned_metrics, indent=2), encoding="utf-8"
    )

    print("Baseline entity F1:", baseline_metrics["overall_f1"], "| token acc:", baseline_metrics["token_accuracy_excl_padding"])
    print("Fine-tuned entity F1:", finetuned_metrics["overall_f1"], "| token acc:", finetuned_metrics["token_accuracy_excl_padding"])
    print("Wrote", REPORTS_DIR / "baseline_metrics.json")
    print("Wrote", REPORTS_DIR / "fine_tuned_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

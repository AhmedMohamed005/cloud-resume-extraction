"""
Test your trained model on a resume text.

Usage:
    python test_model.py

Loads the model from models/resume-ner/final/ and runs it on a sample resume.
Shows each extracted entity with its label and confidence score.
"""
from __future__ import annotations
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel

MODEL_DIR = "models/resume-ner/final"

SAMPLE_RESUME = """
Jackson Miller
Senior React Developer
help@enhancv.com Philadelphia, Pennsylvania

Experience
Accenture Philadelphia, Pennsylvania
Senior React Developer 2018 - Ongoing
Led the frontend development for prominent client projects involving high-traffic web applications.
Collaborated in the design of 5+ innovative UI features, resulting in 15% increase in user engagement.

IBM Philadelphia, Pennsylvania
Mid-Level React Developer 2015 - 2018
Participated in designing and implementing user interface for several strategic projects.
Integrated RESTful services into software solutions, improving response times by 25%.

Education
University of Pennsylvania Philadelphia, Pennsylvania
Master's Degree in Computer Science 2010 - 2012

Pennsylvania State University University Park, Pennsylvania
Bachelor's Degree in Computer Science 2006 - 2010

Skills
React JavaScript TypeScript Redux HTML CSS Agile RESTful APIs
"""


def load_model(model_dir: str):
    label_map_path = Path(model_dir) / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.json not found in {model_dir}")

    with open(label_map_path) as f:
        label_map = json.load(f)

    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = label_map["label2id"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load base model first
    base_model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Then wrap it with the saved LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()  # fuse LoRA weights into base model
    
    return tokenizer, model, id2label


def group_entities(raw_entities: list[dict]) -> list[dict]:
    """
    Group consecutive B-/I- tokens into single entity spans.
    Input:  [{"word": "David", "entity": "B-NAME", ...}, {"word": "Clark", "entity": "I-NAME", ...}]
    Output: [{"text": "David Clark", "label": "NAME", "score": 0.98}]
    """
    if not raw_entities:
        return []

    grouped = []
    current = None

    for token in raw_entities:
        entity_tag = token["entity"]
        score      = token["score"]
        word       = token["word"].replace("##", "")  # remove BERT wordpiece prefix

        if entity_tag.startswith("B-"):
            if current:
                grouped.append(current)
            current = {
                "text":  word,
                "label": entity_tag[2:],
                "score": score,
            }
        elif entity_tag.startswith("I-") and current:
            # Check it's the same entity type
            if entity_tag[2:] == current["label"]:
                # Reconstruct word: if original ends without space, no space needed
                current["text"] += " " + word if not word.startswith("'") else word
                current["score"] = (current["score"] + score) / 2
        else:
            if current:
                grouped.append(current)
                current = None

    if current:
        grouped.append(current)

    return grouped


def main():
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        print(f"Model not found at {MODEL_DIR}")
        print("Run train.py first.")
        return

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer, model, id2label = load_model(MODEL_DIR)

    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=-1,   # CPU; change to 0 for GPU
    )

    print("Running inference on sample resume...\n")
    raw_entities = ner(SAMPLE_RESUME)
    entities = group_entities(raw_entities)

    # Group by label type for clean display
    by_type: dict[str, list] = {}
    for ent in entities:
        by_type.setdefault(ent["label"], []).append(ent)

    print("=" * 55)
    print("EXTRACTED ENTITIES")
    print("=" * 55)
    for label_type in ["NAME", "EXPERIENCE", "COMPANY", "SKILL", "EDUCATION", "DATE", "LOCATION"]:
        items = by_type.get(label_type, [])
        if not items:
            continue
        print(f"\n{label_type}:")
        for item in items:
            conf = f"{item['score']:.2f}"
            print(f"  [{conf}]  {item['text']}")
    print("=" * 55)

    # Also show raw JSON for integration into your API
    print("\nJSON output (for API integration):")
    result = {
        label: [item["text"] for item in items]
        for label, items in by_type.items()
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
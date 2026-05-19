"""LoRA NER model loading and inference (BERT token classification)."""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from app.schemas.output_schema import CandidateProfile
from app.services.inference import (
    _first_email,
    _first_phone,
    _normalize_education_lines,
    extract_name,
    run_mock_inference,
)

logger = logging.getLogger(__name__)

_DEFAULT_REL_MODEL = Path("models") / "resume-ner" / "final"


def _repo_root() -> Path:
    # code/app/services/ner_engine.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3]


def resolve_model_dir() -> Path:
    raw = os.environ.get("RESUME_MODEL_PATH", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
    else:
        p = (_repo_root() / _DEFAULT_REL_MODEL).resolve()
    return p


def _model_ready(model_dir: Path) -> bool:
    return model_dir.is_dir() and (model_dir / "label_map.json").exists()


def _merged_bundle_dir(model_dir: Path) -> Path | None:
    """Prefer standalone merged export (train.py / export_merged_lora.py) over runtime PEFT merge."""
    merged = model_dir / "merged"
    if (merged / "config.json").is_file():
        return merged
    return None


def _ner_doc_peak_min() -> float:
    """If best token prob is below this, treat the doc as 'no confident NER' (rely on heuristics)."""
    return max(0.0, min(1.0, float(os.environ.get("NER_DOC_PEAK_MIN", "0.20"))))


def _ner_score_floor(peak: float) -> float:
    """Keep tokens/spans above max(absolute floor, relative * peak)."""
    abs_floor = float(os.environ.get("NER_ABS_SCORE_FLOOR", "0.17"))
    rel = float(os.environ.get("NER_REL_TO_PEAK", "0.50"))
    return max(abs_floor, rel * peak)


def _filter_raw_ner_tokens(raw: list[dict], peak: float, peak_min: float) -> list[dict]:
    if not raw or peak < peak_min:
        return []
    floor = _ner_score_floor(peak)
    out: list[dict] = []
    for t in raw:
        try:
            if float(t.get("score", 0.0)) >= floor:
                out.append(t)
        except (TypeError, ValueError):
            continue
    return out


def _filter_grouped_spans(entities: list[dict], peak: float, peak_min: float) -> list[dict]:
    if peak < peak_min:
        return []
    floor = _ner_score_floor(peak)
    return [e for e in entities if float(e.get("score", 0.0)) >= floor]


@lru_cache(maxsize=1)
def _load_ner_pipeline():
    model_dir = resolve_model_dir()
    if not _model_ready(model_dir):
        raise FileNotFoundError(f"NER model not found or incomplete at {model_dir}")

    label_map_path = model_dir / "label_map.json"
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)

    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = label_map["label2id"]

    merged_dir = _merged_bundle_dir(model_dir)
    if merged_dir is not None:
        logger.info("Loading merged NER weights from %s", merged_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(merged_dir))
    else:
        logger.info("Loading PEFT adapter from %s (no merged/ bundle)", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        base_model = AutoModelForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        model = model.merge_and_unload()

    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    logger.info("Loaded NER pipeline from %s (device=%s)", model_dir, device)
    return ner


def group_entities(raw_entities: list[dict]) -> list[dict]:
    """Group consecutive B-/I- tokens into single entity spans."""
    if not raw_entities:
        return []

    grouped: list[dict] = []
    current: dict | None = None

    for token in raw_entities:
        entity_tag = token["entity"]
        score = float(token["score"])
        word = str(token["word"]).replace("##", "")

        if entity_tag.startswith("B-"):
            if current:
                grouped.append(current)
            current = {
                "text": word,
                "label": entity_tag[2:],
                "score": score,
            }
        elif entity_tag.startswith("I-") and current:
            if entity_tag[2:] == current["label"]:
                sep = "" if word.startswith("'") else " "
                current["text"] = current["text"] + sep + word
                current["score"] = (current["score"] + score) / 2.0
        else:
            if current:
                grouped.append(current)
                current = None

    if current:
        grouped.append(current)

    return grouped


def _entities_to_profile(entities: list[dict], full_text: str) -> tuple[CandidateProfile, float]:
    by_label: dict[str, list[dict]] = {}
    for ent in entities:
        by_label.setdefault(ent["label"], []).append(ent)

    name = None
    if items := by_label.get("NAME"):
        name = items[0]["text"].strip()
    if not name:
        name = extract_name(full_text)

    skills = []
    for item in by_label.get("SKILL", []):
        t = item["text"].strip()
        if t and t not in skills:
            skills.append(t)

    education = []
    for item in by_label.get("EDUCATION", []):
        t = item["text"].strip()
        if t and t not in education:
            education.append(t)

    experience = []
    for item in by_label.get("EXPERIENCE", []):
        t = item["text"].strip()
        if t and t not in experience:
            experience.append(t)

    email = _first_email(full_text)
    phone = _first_phone(full_text)

    scores: list[float] = []
    for label in ("NAME", "SKILL", "EDUCATION", "EXPERIENCE"):
        for item in by_label.get(label, []):
            scores.append(float(item["score"]))

    confidence = sum(scores) / len(scores) if scores else 0.45

    profile = CandidateProfile(
        name=name,
        email=email,
        phone=phone,
        skills=skills,
        education=education,
        experience=experience,
    )
    return profile, confidence


def run_ner_inference(clean_text: str) -> tuple[CandidateProfile, float, dict[str, Any]]:
    """Token NER via transformers pipeline (truncation handled inside the pipeline)."""
    ner = _load_ner_pipeline()
    # Transformers 5.x TokenClassificationPipeline does not accept `truncation=` here.
    raw = ner(clean_text)
    peak = max((float(t.get("score", 0.0)) for t in raw), default=0.0)
    pmin = _ner_doc_peak_min()
    raw = _filter_raw_ner_tokens(raw, peak, pmin)
    entities = group_entities(raw)
    entities = _filter_grouped_spans(entities, peak, pmin)
    profile, confidence = _entities_to_profile(entities, clean_text)
    detected = {
        "skills": profile.skills,
        "education": profile.education,
        "experience": profile.experience,
        "ner_entities": [{"text": e["text"], "label": e["label"], "score": e["score"]} for e in entities],
    }
    return profile, confidence, detected


def _clean_entity_list(items: list[str]) -> list[str]:
    """Drop punctuation-only / single-letter NER fragments."""
    junk = {":", "-", "–", "—", ".", ",", "n", "oh", "o", "um", "uh", "ha"}
    out: list[str] = []
    for x in items:
        s = x.strip()
        if len(s) < 2 or s.lower() in junk:
            continue
        alnum = sum(1 for c in s if c.isalnum())
        if alnum < 2:
            continue
        out.append(s)
    return out


_EXP_HEADER_PHRASES = frozenset(
    {
        "experience",
        "work experience",
        "professional experience",
        "relevant experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "professional background",
    }
)

_EDU_HEADER_PHRASES = frozenset(
    {
        "education",
        "academic",
        "academics",
        "qualifications",
        "educational background",
        "academic background",
    }
)

_YEAR_ONLY = re.compile(r"^\d{1,4}$")
_YEAR_RANGE_ONLY = re.compile(r"^\s*\d{2,4}\s*[-–—]\s*\d{2,4}\s*$")

# Standalone NER / OCR fragments of longer words (e.g. "utilization" → "zation")
_EXP_JUNK_TAILS = frozenset(
    {"zation", "ization", "isation", "ations", "tions", "ments", "sions", "ology", "ologies"}
)


def _clean_experience_list(items: list[str]) -> list[str]:
    """Remove section headers and year-only NER fragments from EXPERIENCE spans."""
    out: list[str] = []
    for x in _clean_entity_list(items):
        low = x.strip().lower().rstrip(":").strip()
        if low in _EXP_HEADER_PHRASES:
            continue
        t = x.strip()
        if _YEAR_ONLY.match(t) or _YEAR_RANGE_ONLY.match(t):
            continue
        # NER subword junk (e.g. "han", "ge") — short all-lowercase alpha tokens
        if len(t) <= 3 and t.isalpha() and t.islower():
            continue
        if low in {"han", "ge", "um", "uh", "ologies", "metho"}:
            continue
        words = t.split()
        if len(words) == 2 and len(words[0]) == 1 and words[0].isalpha() and len(words[1]) <= 6:
            continue
        if len(t) <= 3 and t.isalpha() and t.isupper() and t in {
            "ES", "US", "IT", "FR", "UK", "HR", "PR", "ER", "DR", "MS",
        }:
            continue
        if low in _EXP_JUNK_TAILS:
            continue
        if len(t) <= 10 and " " not in t and t.islower() and t.endswith(("zation", "ization", "isation")):
            continue
        out.append(t)
    return out


def _clean_education_list(items: list[str]) -> list[str]:
    """Remove common section-header false positives from EDUCATION spans."""
    out: list[str] = []
    for x in _clean_entity_list(items):
        low = x.strip().lower().rstrip(":").strip()
        if low in _EDU_HEADER_PHRASES:
            continue
        out.append(x.strip())
    return out


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def _name_likely_bad(name: str | None) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    low = name.lower()
    if any(b in low for b in ("skills", "experience", "education", "summary", "resume", "http")):
        return True
    if sum(1 for w in name.split() if w.isalpha()) < 2:
        return True
    return False


def run_ner_with_heuristic_merge(clean_text: str) -> tuple[CandidateProfile, float, dict[str, Any]]:
    """
    Run LoRA NER, then fill gaps from the same heuristics used by /extract mock path.
    Ensures usable JSON even when the NER head misses spans or is weakly trained.
    """
    ner_profile, ner_conf, ner_det = run_ner_inference(clean_text)
    mock_profile, mock_conf, mock_sections = run_mock_inference(clean_text)

    use_ner_name = ner_profile.name and not _name_likely_bad(ner_profile.name)
    # Always re-scan full text for contact: OCR can differ from what NER path retained.
    email = _first_email(clean_text) or ner_profile.email or mock_profile.email
    phone = _first_phone(clean_text) or ner_profile.phone or mock_profile.phone
    merged = CandidateProfile(
        name=(ner_profile.name if use_ner_name else None) or mock_profile.name,
        email=email,
        phone=phone,
        skills=_dedupe_preserve_order(
            _clean_entity_list(list(ner_profile.skills)) + list(mock_profile.skills)
        ),
        education=_normalize_education_lines(
            _dedupe_preserve_order(
                _clean_education_list(list(ner_profile.education)) + list(mock_profile.education)
            )
        ),
        experience=_dedupe_preserve_order(
            _clean_experience_list(list(ner_profile.experience)) + list(mock_profile.experience)
        ),
    )
    # Confidence: prefer NER when it emitted any structured entities, else heuristic score
    has_ner = bool(ner_det.get("ner_entities"))
    confidence = max(ner_conf, mock_conf) if has_ner else mock_conf

    detected = {
        **ner_det,
        "detected_sections": mock_sections,
        "merged_with_heuristics": True,
    }
    return merged, confidence, detected


def clear_ner_cache() -> None:
    """For tests that swap RESUME_MODEL_PATH."""
    _load_ner_pipeline.cache_clear()

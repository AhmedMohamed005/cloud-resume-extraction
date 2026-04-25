from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, List

from app.schemas.output_schema import CandidateProfile


_SKILL_BANK = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "cpp", "csharp",
    "go", "rust", "swift", "kotlin", "ruby", "php", "scala", "r",
    "bash", "shell", "html", "css", "sql", "graphql", "jquery",
    # Frameworks / libraries
    "react", "reactjs", "angular", "vue", "vuejs", "nextjs", "nuxtjs",
    "node", "nodejs", "express", "fastapi", "django", "flask", "spring",
    "rails", "laravel", "svelte", "redux", "mobx", "enzyme", "jest",
    "webpack", "vite", "babel",
    # Data / ML
    "pytorch", "tensorflow", "keras", "scikit", "pandas", "numpy",
    "spark", "kafka", "airflow", "dbt", "mlflow",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "githubactions", "circleci", "helm",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "sqlite", "dynamodb", "cassandra", "neo4j",
    # Tools
    "git", "linux", "nginx", "graphql", "rest", "grpc",
}
 


_NAME_STOPWORDS = {
    "summary", "objective", "resume", "curriculum vitae", "cv",
    "profile", "contact", "personal information", "skills",
    "experience", "education", "references", "projects",
    "certifications", "achievements",
}

_SECTION_HEADERS = {
    "summary", "objective", "skills", "experience", "education",
    "projects", "certifications", "achievements", "references",
    "employment", "work history", "academic", "languages", "tools",
}

_STOP_WORDS = {"experience", "skills", "projects", "work"}

_PHONE_PATTERNS = [
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\d)"),
]


def _first_email(text: str):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def _first_phone(text: str):
    flat_text = re.sub(r"\s+", " ", text)

    for pattern in _PHONE_PATTERNS:
        for m in pattern.finditer(flat_text):
            candidate = m.group(0).strip()

            # remove weird line breaks / spaces
            candidate = candidate.replace("\n", " ").strip()

            # extract only digits
            digits = re.sub(r"\D", "", candidate)

            # validate length
            if not (10 <= len(digits) <= 15):
                continue

            # 🔥 FIX: take last 10-11 digits (real phone)
            if len(digits) > 10:
                digits = digits[-10:]

            # format nicely
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

            return formatted

    return None


def score_name_candidate(line: str) -> int:
    score = 0
    words = line.split()

    if 2 <= len(words) <= 4:
        score += 2

    if any(char.isdigit() for char in line):
        score -= 3

    low = line.lower()
    if "@" in line or "http" in low:
        score -= 3

    if line.istitle():
        score += 2

    if len(line) < 40:
        score += 1

    normalized = re.sub(r"[^a-z\s]", "", low).strip()
    if normalized in _NAME_STOPWORDS or normalized in _SECTION_HEADERS:
        score -= 5

    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-.]*", w) for w in words if w):
        score -= 2

    return score


def extract_name(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = lines[:10]
    if not candidates:
        return None

    best = max(candidates, key=score_name_candidate, default="")
    if not best or score_name_candidate(best) <= 0:
        return None
    return " ".join(w.capitalize() for w in best.split())


def _name_candidates(text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = lines[:10]
    return [{"line": c, "score": score_name_candidate(c)} for c in candidates]


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _extract_skills(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    found = set()

    for skill in _SKILL_BANK:
        skill_tokens = skill.split()
        token_len = len(skill_tokens)

        for i in range(len(tokens)):
            window = " ".join(tokens[i:i + token_len])
            if similar(skill, window) > 0.8:
                found.add(skill)

    return sorted(found)


def _extract_education(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    education_section: list[str] = []
    capture = False

    for line in lines:
        low = line.lower()

        if "education" in low or "academic" in low:
            capture = True
            continue

        if capture:
            if any(stop in low for stop in _STOP_WORDS):
                break
            if line:
                education_section.append(line)

    return education_section


def _section_lines(text: str, header_keywords: list[str]) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    capture = False
    results: list[str] = []

    for line in lines:
        low = line.lower()
        if any(k in low for k in header_keywords):
            capture = True
            continue
        if capture and low in {"skills", "education", "experience", "projects", "summary"}:
            break
        if capture and line:
            results.append(line)
            if len(results) >= 5:
                break
    return results


def debug_output(parsed_text: str, clean_text: str, profile: CandidateProfile) -> dict[str, Any]:
    return {
        "raw_sample": parsed_text[:500],
        "clean_sample": clean_text[:500],
        "lines": clean_text.split("\n")[:20],
        "name_candidates": _name_candidates(clean_text),
        "profile": profile.model_dump(),
    }


def run_mock_inference(clean_text: str) -> tuple[CandidateProfile, float]:
    profile = CandidateProfile(
        name=extract_name(clean_text),
        email=_first_email(clean_text),
        phone=_first_phone(clean_text),
        skills=_extract_skills(clean_text),
        education=_extract_education(clean_text),
        experience=_section_lines(clean_text, ["experience", "employment", "work history"]),
    )

    # Placeholder confidence for pre-model integration stage.
    confidence = 0.62 if clean_text else 0.0
    return profile, confidence

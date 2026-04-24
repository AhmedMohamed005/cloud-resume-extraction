from __future__ import annotations

import re
from typing import List

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


def _is_likely_name(line: str) -> bool:
    cleaned = line.strip()
    if not cleaned or "@" in cleaned or any(ch.isdigit() for ch in cleaned):
        return False

    low = re.sub(r"[^a-z\s]", "", cleaned.lower()).strip()
    if not low or low in _NAME_STOPWORDS:
        return False

    words = cleaned.split()
    if not 2 <= len(words) <= 4:
        return False

    if len(cleaned) > 50:
        return False

    # Permit common name punctuation but reject noisy lines.
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-.]*", w) for w in words):
        return False

    return True


def _guess_name(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for line in lines[:20]:
        if any(k in line.lower() for k in ["summary", "experience", "education"]):
            continue

        # ignore noisy lines
        if len(line) > 40:
            continue

        if re.search(r"\d|@|http", line):
            continue

        words = line.split()
        if 2 <= len(words) <= 4:
            return " ".join(w.capitalize() for w in words)

    return None

def _extract_skills(text: str) -> List[str]:
    found = set()
    tokens = re.findall(r"[A-Za-z][A-Za-z+.#-]{1,30}", text.lower())
    for token in tokens:
        if token in _SKILL_BANK:
            found.add(token)
    return sorted(found)


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


def run_mock_inference(clean_text: str) -> tuple[CandidateProfile, float]:
    profile = CandidateProfile(
        name=_guess_name(clean_text),
        email=_first_email(clean_text),
        phone=_first_phone(clean_text),
        skills=_extract_skills(clean_text),
        education=_section_lines(clean_text, ["education", "academic"]),
        experience=_section_lines(clean_text, ["experience", "employment", "work history"]),
    )

    # Placeholder confidence for pre-model integration stage.
    confidence = 0.62 if clean_text else 0.0
    return profile, confidence

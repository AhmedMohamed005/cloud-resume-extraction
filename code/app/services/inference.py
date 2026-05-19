from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, List

from app.schemas.output_schema import CandidateProfile


# ── Skill bank ────────────────────────────────────────────────────────────────
# Added: t-sql, tsql, data migration, and many others that were missing.
# Rule: every entry must be lowercase. Matching is done lowercase → lowercase.
_SKILL_BANK = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "cpp", "csharp",
    "go", "rust", "swift", "kotlin", "ruby", "php", "scala", "r",
    "bash", "shell", "html", "css", "sql", "graphql", "jquery",
    # SQL variants — this was the main gap for the David Clark resume
    "t-sql", "tsql", "pl/sql", "plsql", "nosql",
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
    # Tools / practices
    "git", "linux", "nginx", "rest", "grpc",
    "data migration", "database design", "database optimization",
    "data modeling", "etl", "data warehousing",
    "stakeholder management", "audit", "reconciliation",
    "information technology",
    # Frontend / UI
    "react", "reactjs", "angular", "vue", "vuejs", "nextjs",
    "redux", "typescript", "javascript", "html", "css",
    "ui/ux", "ux", "ui", "figma", "tailwind", "sass",
    # Methodologies
    "agile", "scrum", "kanban", "ci/cd", "tdd", "bdd",
    # APIs / communication
    "rest", "restful", "graphql", "grpc", "websocket",
    "api", "soap", "oauth",
    # Additional cloud / tools
    "aws", "azure", "gcp", "s3", "ec2", "lambda",
    "terraform", "ansible", "kubernetes", "docker",
    # ERP / business / marketing (common in non-dev resumes)
    "sap", "abap", "crm", "salesforce", "hubspot", "marketing",
    "digital marketing", "telemarketing", "sales", "erp",
    "bentley", "autodesk", "projectwise", "kpi", "kpis",
    "qa", "quality assurance", "business intelligence", "power bi",
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
    # FIX: add the sections that were not stopping the education capture
    "strengths", "passions", "certification", "interests", "awards",
    "volunteering", "publications", "hobbies",
}

_PHONE_PATTERNS = [
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\d)"),
]

# Egypt mobile: 01[0125] + 8 digits (11 total), often written without spaces
_EGYPT_MOBILE_RE = re.compile(r"(?<!\d)(01[0125]\d{8})(?!\d)")
# Egypt international +20 + 10-digit national (e.g. +201094745504)
_EGYPT_INTL_RE = re.compile(r"(?<!\d)\+20\s*([1-9]\d{9})(?!\d)")

# "Porsche Costa Mesa, Ca" / "Ga Telesis Fort Lauderdale, FL" — employer + city + state, not a person name
_US_STATE_2 = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
)


def _looks_like_company_city_state(line: str) -> bool:
    """Heuristic: 4+ tokens and (multi-part address OR trailing US-style state)."""
    s = line.strip()
    words = s.split()
    if len(words) < 4:
        return False
    if s.count(",") >= 2:
        return True
    m = re.search(r",\s*([A-Za-z]{1,3})\s*$", s)
    if not m:
        return False
    tail = m.group(1)
    if len(tail) == 2 and tail.upper() in _US_STATE_2:
        return True
    if len(tail) == 2 and tail[0].isupper() and tail[1].islower():
        return True
    return False


def _first_email(text: str):
    """
    Best-effort email from resume text. Handles OCR spacing around @ and 'at' spell-outs.
    """
    if not text:
        return None
    # Tighten OCR-split @: "user @ domain.com" already normalized in preprocess; repeat safe here
    blob = re.sub(r"\s*@\s*", "@", text.replace("\uff20", "@"))
    strict = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    m = re.search(strict, blob)
    if m:
        return m.group(0)
    # Missing @: "user domain.com" on one line (OCR dropped @)
    m = re.search(
        r"\b([A-Za-z0-9._%+-]{2,40})\s+([A-Za-z0-9][A-Za-z0-9.-]*\.(?:com|net|org|edu|io|co|gov|uk))\b",
        blob,
        re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    # "user at domain.com"
    m = re.search(
        r"\b([A-Za-z0-9._%+-]{2,40})\s+at\s+([A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,})\b",
        blob,
        re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    return None


def _format_phone_digits(digits: str) -> str | None:
    """Format digit-only string; avoid turning Egyptian 01… into fake US (115) numbers."""
    if not digits.isdigit() or len(digits) < 10:
        return None
    if len(digits) == 11 and digits.startswith("01") and digits[2] in "0125":
        return f"{digits[:3]} {digits[3:7]} {digits[7:]}"
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits[0] == "1" and digits[1] in "3456789":
        return _format_phone_digits(digits[1:])
    if 11 <= len(digits) <= 15:
        return f"+{digits}"
    return None


def _first_phone(text: str):
    # +20 + 10 digits (do not merge whole-resume digits — that breaks on emails / dates)
    norm = re.sub(r"[\u200b-\u200d\ufe0f\u00a0]", " ", text)
    m = _EGYPT_INTL_RE.search(norm)
    if m:
        ten = m.group(1)
        return f"+20 {ten[:2]} {ten[2:6]} {ten[6:]}"

    compact = re.sub(r"\s+", "", text)
    m = _EGYPT_MOBILE_RE.search(compact)
    if m:
        formatted = _format_phone_digits(m.group(1))
        if formatted:
            return formatted

    flat_text = re.sub(r"\s+", " ", text)
    for pattern in _PHONE_PATTERNS:
        for match in pattern.finditer(flat_text):
            candidate = match.group(0).strip().replace("\n", " ").strip()
            digits = re.sub(r"\D", "", candidate)
            if len(digits) == 11 and digits.startswith("01") and digits[2] in "0125":
                return _format_phone_digits(digits)
            if not (10 <= len(digits) <= 15):
                continue
            if len(digits) > 10 and not (digits.startswith("01") or (digits[0] == "1" and digits[1] in "3456789")):
                continue
            if len(digits) > 10 and digits[0] == "1" and digits[1] in "3456789":
                digits = digits[1:]
            elif len(digits) > 10:
                continue
            formatted = _format_phone_digits(digits)
            if formatted:
                return formatted
    return None


def score_name_candidate(line: str) -> int:
    score = 0
    words = line.split()
    low = line.lower()

    # Word count: names are 2-4 words
    if 2 <= len(words) <= 4:
        score += 2
    elif len(words) > 4:
        score -= 2

    # Digits / special chars
    if any(char.isdigit() for char in line):
        score -= 3
    if "@" in line or "http" in low:
        score -= 3

    # Capitalisation signals
    if line.isupper() and 2 <= len(words) <= 4:
        score += 3   # ALL CAPS full name — very common (e.g. "JESSICA CLAIRE")
    elif line.istitle():
        score += 2   # Title Case

    # Length
    if len(line) < 40:
        score += 1

    if _looks_like_company_city_state(line):
        score -= 10

    # Stopword / section header match
    normalized = re.sub(r"[^a-z\s]", "", low).strip()
    if normalized in _NAME_STOPWORDS or normalized in _SECTION_HEADERS:
        score -= 5

    # Skill/soft-skill phrase penalty
    # Lines like "Languages Object-Oriented Programming" contain non-name keywords
    _SKILL_PHRASE_WORDS = {
        "programming", "development", "management", "integration",
        "languages", "technical", "skills", "oriented", "analytical",
        "thinking", "evaluation", "optimization", "methodology",
    }
    word_set = {w.lower() for w in words}
    if word_set & _SKILL_PHRASE_WORDS:
        score -= 4

    # All words must look like name tokens (letters, apostrophes, hyphens only)
    if not all(re.fullmatch(r"[A-Za-z][a-z'\-]*", w) or w.isupper()
               for w in words if w):
        score -= 2

    return score


def extract_name(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = lines[:15]  # extended from 10 — some resumes have address before name
    if not candidates:
        return None
    scored = [(ln, score_name_candidate(ln)) for ln in candidates]
    scored.sort(key=lambda x: -x[1])
    for line, sc in scored:
        if sc <= 0:
            break
        if _looks_like_company_city_state(line):
            continue
        return " ".join(w.capitalize() for w in line.split())
    return None


def _name_candidates(text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [{"line": c, "score": score_name_candidate(c)} for c in lines[:10]]


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ── OCR noise filter ──────────────────────────────────────────────────────────
# Lines that are clearly OCR artifacts: very short, no real words, or
# start with a stray icon character (e.g. "jo", "wv", "5 ", "@8").
_OCR_NOISE_RE = re.compile(
    r"""
    ^(                          # start of line
        \W+                     |  # only symbols / punctuation
        [a-z]{1,2}\s+\w.*       |  # 1-2 char prefix then text → icon OCR artifact
        \d+\s*[a-zA-Z]{0,2}$   |  # lone number or number + 1-2 letters
        [\W\d]{1,4}$               # very short garbage
    )$
    """,
    re.VERBOSE,
)

# Matches lines that are purely a date range + location, e.g.:
#   "#8 2003-2007 9 Columbus, OH"   "@8 2017-Ongoing 9 Columbus, OH"
#   "68 2013-2017 9 Columbus, OH"
_DATE_LOCATION_RE = re.compile(
    r"^[\W\d]{0,4}\s*\d{4}\s*[-–]\s*(\d{4}|ongoing|present|now|current)"
    r".*$",
    re.IGNORECASE,
)

# Matches contact-info lines that leak into sections, e.g.:
#   "@ help@enhancv.com @ linkedin.com"   "9 Columbus, Ohio"
_CONTACT_LINE_RE = re.compile(
    r"(@.*@|linkedin\.com|github\.com|twitter\.com|^\s*[\W]{0,3}\s*\w+,\s*\w+\s*$)",
    re.IGNORECASE,
)


def _is_ocr_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    if _OCR_NOISE_RE.match(stripped):
        return True
    # Lines that are ONLY non-ASCII characters (icon fonts rendered as junk)
    ascii_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
    if ascii_chars == 0:
        return True
    # Date + location lines like "#8 2003-2007 9 Columbus, OH" (not plain "2023 - Present")
    if _DATE_LOCATION_RE.match(stripped):
        if re.match(
            r"^\d{4}\s*[-–—]?\s*(present|current|now|ongoing)\s*$",
            stripped,
            re.IGNORECASE,
        ):
            return False
        return True
    # Contact info lines that bleed into sections
    if _CONTACT_LINE_RE.search(stripped) and "@" in stripped:
        return True
    return False


def _clean_section_lines(lines: list[str], drop_states: bool = False) -> list[str]:
    """Remove OCR noise lines and deduplicate.

    drop_states=True: also remove standalone US state continuation lines
    (used for education section where city appears on one line, state on next).
    """
    seen = set()
    result = []
    for line in lines:
        if _is_ocr_noise(line):
            continue
        if drop_states and _is_standalone_state(line):
            continue
        key = re.sub(r"\s+", " ", line.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        result.append(line.strip())
    return result



# US state names that appear as standalone continuation lines in city/state addresses
# e.g. "University of Pennsylvania Philadelphia," followed by "Pennsylvania" alone
_US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
}

def _is_standalone_state(line: str) -> bool:
    """Return True if line is just a US state name (city/state continuation line)."""
    return line.strip().lower() in _US_STATES


_EDU_DATE_PREFIX = re.compile(
    r"^(\d{4})\s*[-–—]?\s*(present|current|now|presen|ongoing)\.?\s*$",
    re.IGNORECASE,
)


def _normalize_education_lines(lines: list[str]) -> list[str]:
    """Merge '2023 – Present' + degree line; drop orphan year-only fragments."""
    if not lines:
        return []
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _EDU_DATE_PREFIX.match(line) and nxt:
            out.append(f"{line} {nxt}")
            i += 2
            continue
        if _EDU_DATE_PREFIX.match(line) and not nxt:
            i += 1
            continue
        if re.match(r"^\d{4}\s*[-–—]?\s*$", line) and nxt:
            out.append(f"{line.strip()} {nxt}")
            i += 2
            continue
        out.append(line)
        i += 1
    return out


# ── Section splitting ─────────────────────────────────────────────────────────
# FIX: expanded SECTION_MAP to include all headers seen in real resumes.
# This prevents sections like STRENGTHS / PASSIONS from being swallowed
# into the education bucket.
_SECTION_MAP: dict[str, list[str]] = {
    "experience":  [
        "experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "work experience",
        "relevant experience",
        "internship experience",
    ],
    "education":   ["education", "academic background", "academic"],
    "skills":      ["skills", "technical skills", "core skills", "key skills",
                    "competencies", "technologies", "technical skill"],
    # These don't map to schema fields but must be recognised as section
    # boundaries so they STOP the previous section from over-capturing.
    "_strengths":  ["strengths"],
    "_passions":   ["passions", "interests", "hobbies"],
    "_certifications": ["certification", "certifications", "licenses"],
    "_awards":     ["awards", "achievements", "accomplishments"],
    "_projects":   ["projects", "personal projects", "portfolio"],
    "_summary":    ["summary", "professional summary", "profile", "objective", "about me"],
    "_references": ["references"],
}


def split_sections(text: str) -> dict[str, list[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: dict[str, list[str]] = {}
    current = "other"   # lines before the first recognised header
    sections[current] = []

    for line in lines:
        low = line.lower().strip()
        matched = False
        for sec, keywords in _SECTION_MAP.items():
            for k in keywords:
                if low == k:
                    current = sec
                    sections.setdefault(current, [])
                    matched = True
                    break
                head, _, tail = line.partition(":")
                if head.strip().lower() == k:
                    current = sec
                    sections.setdefault(current, [])
                    if tail.strip():
                        sections[current].append(tail.strip())
                    matched = True
                    break
                if low.startswith(k):
                    current = sec
                    sections.setdefault(current, [])
                    matched = True
                    break
            if matched:
                break
        if not matched:
            sections.setdefault(current, []).append(line)

    # "other" = pre-header block (name, title, contact info).
    # Never merge it into experience/education — discard it.
    sections.pop("other", None)

    # Clean each section's lines
    # Education gets extra cleanup: standalone state-name continuation lines removed
    for key in list(sections.keys()):
        sections[key] = _clean_section_lines(
            sections[key],
            drop_states=(key == "education"),
        )
        if key == "education":
            sections[key] = _normalize_education_lines(sections[key])

    return sections


# ── Skill extraction ──────────────────────────────────────────────────────────
# FIX: search the FULL resume text, not just the skills section lines.
# Many resumes mention skills throughout the experience section.
# We now do two passes:
#   1. Exact match against the full text (fast, handles multi-word skills)
#   2. Fuzzy match on individual tokens (catches typos like "Postgre sql")

def _extract_skills(full_text: str, skills_section_text: str = "") -> List[str]:
    """
    Extract skills from the skills section first, then supplement with
    mentions anywhere in the full resume text.
    """
    search_text = (skills_section_text + " " + full_text).lower()
    # Normalise: collapse whitespace so multi-word skills match across line breaks
    search_text = re.sub(r"\s+", " ", search_text)

    found: set[str] = set()

    for skill in _SKILL_BANK:
        skill_lower = skill.lower()
        skill_tokens = skill_lower.split()
        token_len = len(skill_tokens)

        if token_len == 1:
            # Single-word: use word-boundary regex for speed + accuracy
            pattern = rf"\b{re.escape(skill_lower)}\b"
            if re.search(pattern, search_text):
                found.add(skill)
        else:
            # Multi-word: sliding window with fuzzy match
            tokens = search_text.split()
            for i in range(len(tokens) - token_len + 1):
                window = " ".join(tokens[i : i + token_len])
                if similar(skill_lower, window) > 0.85:
                    found.add(skill)
                    break

    return sorted(found)


# ── Main inference entry point ────────────────────────────────────────────────

def run_mock_inference(clean_text: str) -> tuple[CandidateProfile, float, dict]:
    sections = split_sections(clean_text)

    # Skills: search skills section AND full text
    skills_section_lines = sections.get("skills", [])
    skills = _extract_skills(
        full_text=clean_text,
        skills_section_text=" ".join(skills_section_lines),
    )

    profile = CandidateProfile(
        name=extract_name(clean_text),
        email=_first_email(clean_text),
        phone=_first_phone(clean_text),
        skills=skills,
        education=sections.get("education", []),
        experience=sections.get("experience", []),
    )

    confidence = 0.62 if clean_text else 0.0
    detected = {
        "skills": profile.skills,
        "education": profile.education,
        "experience": profile.experience,
    }
    return profile, confidence, detected


# ── Debug helper ──────────────────────────────────────────────────────────────

def debug_output(parsed_text: str, clean_text: str, profile: CandidateProfile) -> dict[str, Any]:
    return {
        "raw_sample": parsed_text[:500],
        "clean_sample": clean_text[:500],
        "lines": clean_text.split("\n")[:20],
        "name_candidates": _name_candidates(clean_text),
        "profile": profile.model_dump(),
    }
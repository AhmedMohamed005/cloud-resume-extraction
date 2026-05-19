from __future__ import annotations

from app.schemas.output_schema import CandidateProfile


def sanitize_profile(profile: CandidateProfile) -> CandidateProfile:
    def clean_items(items: list[str]) -> list[str]:
        out = []
        seen = set()
        for item in items:
            value = item.strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    profile.skills = clean_items(profile.skills)
    profile.education = clean_items(profile.education)
    profile.experience = clean_items(profile.experience)
    return profile

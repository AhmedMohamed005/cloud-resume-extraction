from app.services.inference import run_mock_inference, split_sections


def test_name_heading_is_not_selected_as_candidate_name() -> None:
    text = """
SUMMARY
John Doe
Email: john@example.com
(555) 432-1000
""".strip()

    profile, _, _ = run_mock_inference(text)
    assert profile.name == "John Doe"


def test_professional_experience_header_populates_experience_section() -> None:
    text = """
Skills
Python
Professional Experience
Backend Developer | Startup
Built APIs with Django.
Projects
Side project
""".strip()
    sections = split_sections(text)
    assert any("Backend Developer" in ln for ln in sections.get("experience", []))


def test_name_skips_employer_city_state_line() -> None:
    text = """
Maria Rodriguez
Porsche Costa Mesa, Ca
Marketing and SAP experience
resumesample@example.com
""".strip()
    profile, _, _ = run_mock_inference(text)
    assert profile.name == "Maria Rodriguez"


def test_egypt_international_plus_twenty() -> None:
    text = """
Ahmed Mohamed
📞 +201094745504
ahmed@outlook.com
""".strip()
    profile, _, _ = run_mock_inference(text)
    assert profile.phone == "+20 10 9474 5504"


def test_egypt_mobile_format_not_us_truncated() -> None:
    text = """
Youssef Nasser
youssef@example.com
Egypt
01150306178
""".strip()
    profile, _, _ = run_mock_inference(text)
    assert profile.phone == "011 5030 6178"


def test_education_merges_date_line_with_degree() -> None:
    text = """
Education
2023 – Present
Computer Science
Helwan University
Skills
Python
""".strip()
    profile, _, _ = run_mock_inference(text)
    assert any("2023" in e and "Computer Science" in e for e in profile.education)
    assert any("Helwan" in e for e in profile.education)


def test_phone_extraction_ignores_zip_prefixes() -> None:
    text = """
Contact
94105
(555) 432-1000
john@example.com
""".strip()

    profile, _, _ = run_mock_inference(text)
    assert profile.phone == "(555) 432-1000"

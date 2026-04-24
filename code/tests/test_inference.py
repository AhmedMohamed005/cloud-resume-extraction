from app.services.inference import run_mock_inference


def test_name_heading_is_not_selected_as_candidate_name() -> None:
    text = """
SUMMARY
John Doe
Email: john@example.com
(555) 432-1000
""".strip()

    profile, _ = run_mock_inference(text)
    assert profile.name == "John Doe"


def test_phone_extraction_ignores_zip_prefixes() -> None:
    text = """
Contact
94105
(555) 432-1000
john@example.com
""".strip()

    profile, _ = run_mock_inference(text)
    assert profile.phone == "(555) 432-1000"

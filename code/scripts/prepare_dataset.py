import os
from app.services.parser import extract_text_from_pdf_bytes
from app.services.preprocess import clean_resume_text

INPUT_DIR = "../dataset/Resumes PDF/SQL"
OUTPUT_DIR = "../cleaned"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("dataset/Resumes PDF/React -> cleaned/ (ensure this is correct before running!) dir : ", os.listdir(INPUT_DIR))
def process_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    parsed = extract_text_from_pdf_bytes(pdf_bytes)
    cleaned = clean_resume_text(parsed.text)

    return cleaned


for file in os.listdir(INPUT_DIR):
    if not file.endswith(".pdf"):
        continue

    path = os.path.join(INPUT_DIR, file)
    cleaned_text = process_pdf(path)

    out_path = os.path.join(OUTPUT_DIR, file.replace(".pdf", ".txt"))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[OK] {file}")
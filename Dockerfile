# Resume extraction API — run from repository root:
#   docker build -t resume-extraction .
#   docker run -p 8000:8000 -e USE_MOCK_INFERENCE=0 resume-extraction
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-min.txt /app/requirements-min.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r /app/requirements-min.txt

COPY code/ /app/code/

ENV PYTHONPATH=/app/code
ENV RESUME_MODEL_PATH=/app/models/resume-ner/final
ENV MODEL_S3_URI=
ENV USE_MOCK_INFERENCE=0

WORKDIR /app/code
EXPOSE 8000
CMD ["sh", "-c", "python /app/code/scripts/bootstrap_model.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

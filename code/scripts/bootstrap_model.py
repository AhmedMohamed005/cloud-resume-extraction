"""Prepare the resume model directory before the API starts.

If ``MODEL_S3_URI`` is set to an ``s3://bucket/prefix`` value, the script
downloads that prefix into ``RESUME_MODEL_PATH``. If the local model directory
already exists, the script leaves it untouched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3


@dataclass(frozen=True)
class S3Location:
    bucket: str
    prefix: str


def parse_s3_uri(uri: str) -> S3Location:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri!r}. Expected s3://bucket/prefix")
    prefix = parsed.path.lstrip("/")
    return S3Location(bucket=parsed.netloc, prefix=prefix)


def local_model_dir() -> Path:
    raw = os.environ.get("RESUME_MODEL_PATH", "/app/models/resume-ner/final").strip()
    return Path(raw).expanduser().resolve()


def model_is_ready(model_dir: Path) -> bool:
    return (model_dir / "label_map.json").is_file()


def download_prefix(uri: str, target_dir: Path) -> None:
    location = parse_s3_uri(uri)
    client = boto3.client("s3")

    target_dir.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=location.bucket, Prefix=location.prefix)

    downloaded = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            relative_key = key[len(location.prefix) :].lstrip("/") if location.prefix else key
            if not relative_key:
                continue

            destination = target_dir / relative_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(location.bucket, key, str(destination))
            downloaded += 1

    if downloaded == 0:
        raise FileNotFoundError(f"No files found at {uri}")


def main() -> None:
    model_dir = local_model_dir()
    if model_is_ready(model_dir):
        print(f"Model already present at {model_dir}")
        return

    model_s3_uri = os.environ.get("MODEL_S3_URI", "").strip()
    if not model_s3_uri:
        raise FileNotFoundError(
            f"Model not found at {model_dir} and MODEL_S3_URI is not set"
        )

    print(f"Downloading model from {model_s3_uri} to {model_dir}...")
    download_prefix(model_s3_uri, model_dir)
    if not model_is_ready(model_dir):
        raise FileNotFoundError(
            f"Downloaded model is incomplete: {model_dir} does not contain label_map.json"
        )
    print(f"Model ready at {model_dir}")


if __name__ == "__main__":
    main()
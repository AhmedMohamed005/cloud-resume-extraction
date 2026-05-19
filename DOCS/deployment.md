# Deployment (Docker + AWS EC2)

For **Postman against a public URL**, **Colab tunnels**, and **cURL**, see [POSTMAN_AND_REMOTE_TESTING.md](POSTMAN_AND_REMOTE_TESTING.md).

## Docker (local or EC2)

From the **repository root** (where the `Dockerfile` lives):

```bash
docker build -t resume-extraction .
docker run -d --name resume-api -p 8000:8000 \
  -e USE_MOCK_INFERENCE=0 \
  -e RESUME_MODEL_PATH=/app/models/resume-ner/final \
  -e MODEL_S3_URI=s3://your-bucket/models/resume-ner/final \
  resume-extraction
```

**Environment variables (common)**

| Variable | Example | Notes |
|----------|---------|--------|
| `USE_MOCK_INFERENCE` | `0` | `1` skips PyTorch NER (smaller image testing only). |
| `RESUME_MODEL_PATH` | `/app/models/resume-ner/final` | Must contain `label_map.json` and adapter or use baked-in `merged/`. |
| `MODEL_S3_URI` | `s3://your-bucket/models/resume-ner/final` | Optional. If set, the container downloads the model at startup. |

**Model not inside the image:** choose one of these:

```bash
# Option 1: bind-mount a local copy
docker run -d -p 8000:8000 \
  -v /path/on/host/models/resume-ner/final:/app/models/resume-ner/final:ro \
  -e RESUME_MODEL_PATH=/app/models/resume-ner/final \
  resume-extraction

# Option 2: download from S3 at container start
docker run -d -p 8000:8000 \
  -e MODEL_S3_URI=s3://your-bucket/models/resume-ner/final \
  -e RESUME_MODEL_PATH=/app/models/resume-ner/final \
  resume-extraction
```

The image installs **Tesseract** for OCR (see `Dockerfile`). For AWS, the cleanest path is to keep the model in S3, give the EC2 instance an IAM role that can read that bucket prefix, and let the container download the model on startup. On Windows dev without Tesseract, scanned PDFs may fail locally while succeeding in the container.

**Image dependencies:** `requirements-min.txt` is used in the Dockerfile for a slimmer install than full `requirements.txt`.

## AWS EC2 (recommended course path)

1. **Launch** an instance (Ubuntu 22.04 LTS or Amazon Linux 2023), t3.medium or larger if running PyTorch on CPU (GPU instance if you use CUDA).
2. **Security group:** inbound **TCP 8000** from your IP (or `0.0.0.0/0` for a public class demo — tighten for production). Optionally **22** for SSH from your IP only.
3. **SSH** into the instance; install Docker: `sudo apt update && sudo apt install -y docker.io` (Ubuntu) and `sudo usermod -aG docker $USER` (re-login).
4. **Copy** the repo (without `dataset/` if huge): `git clone …` or `scp` / SFTP.
5. **Upload the model to S3** if you have not already done so: put the contents of `models/resume-ner/final/` under a prefix like `s3://your-bucket/models/resume-ner/final/`.
6. **Attach an IAM role** to the EC2 instance with `s3:GetObject` and `s3:ListBucket` limited to that bucket/prefix.
7. **Build** on the instance: `docker build -t resume-extraction .` from repo root.
8. **Run** with `-p 8000:8000` and `-e MODEL_S3_URI=s3://your-bucket/models/resume-ner/final`. **Public URL:** `http://<Elastic_IP_or_Public_DNS>:8000`.
9. **Verify:** `curl http://<IP>:8000/health` → `{"status":"ok"}`.

**Elastic IP:** Allocate and associate so the IP does not change after stop/start.

### IAM and CloudWatch (course narrative)

- **IAM:** Instance role with minimal permissions; if you add S3 later, attach a policy scoped to one bucket prefix.
- **CloudWatch:** Use the **awslogs** log driver for `docker run`, or install the CloudWatch agent on the host; create alarms on CPU and **5xx** (if behind ALB).

### TLS (optional)

For HTTPS, put **nginx** or **AWS ALB** in front, terminate TLS on **443**, and forward to the container on **8000**. Update Postman `baseUrl` to `https://…`.

## Health checks

Target `GET /health` from a load balancer or uptime monitor.

## Secrets

Do not commit `.env`. Prefer **IAM roles** on EC2 for AWS API access instead of long-lived keys on disk.

## Google Cloud (optional alternative)

**Compute Engine:** Same pattern as EC2 (VM, firewall rule for port 8000, Docker). **Cloud Run:** containerized HTTP service with different scaling and image-size constraints — viable but not the primary documented path for this course repo.

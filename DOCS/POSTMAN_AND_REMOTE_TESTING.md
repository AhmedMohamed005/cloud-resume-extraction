# Postman and remote testing

Use this after Docker or cloud deploy so the same collection works against **any base URL**.

## Postman

1. Import [code/postman/Resume_Extraction.postman_collection.json](../code/postman/Resume_Extraction.postman_collection.json).
2. Open the collection **Variables** (or edit the collection JSON).
3. Set **`baseUrl`** to:
   - Local: `http://127.0.0.1:8000`
   - EC2: `http://<PUBLIC_IP_OR_DNS>:8000` (same port you exposed, e.g. `8000`).
4. **Extract PDF:** `POST {{baseUrl}}/extract`, body **form-data**, key **`file`**, type **File**, choose a `.pdf`.
5. Optional: **`GET {{baseUrl}}/health`** before extract to confirm reachability.

**HTTPS:** If you terminate TLS on nginx/ALB, use `https://...` and ensure the security group / firewall allows **443** instead of (or in addition to) **8000**.

## cURL (sanity check)

```bash
curl -s "http://127.0.0.1:8000/health"
curl -s -F "file=@sample.pdf" "http://127.0.0.1:8000/extract"
```

## Google Colab (short-lived demo only)

Colab is **not** a substitute for EC2 in a “cloud deployment” course narrative, but it can demo the API over the public internet:

1. Install dependencies (same stack as `requirements.txt` or `requirements-min.txt` — expect large downloads for torch).
2. Upload or clone the repo; ensure `models/resume-ner/final/` (or `merged/`) is available.
3. Start the server in the background, e.g. `uvicorn` with `host="0.0.0.0"` and `port=8000`.
4. Expose port **8000** with **ngrok** or **localtunnel** and use the generated HTTPS URL as `baseUrl` in Postman.

**Caveats:** VM resets lose state; cold start and bandwidth; do not rely on Colab for production or for supervisor “persistent cloud” evidence unless the course explicitly allows it.

## Google Cloud (one paragraph)

For parity with “Google” hosting: **Cloud Run** can run a containerized FastAPI app with HTTP ingress, but large ML images and CPU/GPU limits need planning. **Compute Engine** is closer to EC2 (VM + open port + Docker). This repo’s primary documented path remains **Docker + EC2** per the concept note; adapt firewall rules and URLs analogously.

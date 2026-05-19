# MVP storage decision (proposal alignment)

The proposal template (§5 / §8) mentions **SQLite** and **S3** for uploads or results. For this repository’s **course / EC2 MVP**:

## SQLite decision

- **Default:** No SQLite — the API is **stateless**; JSON is the deliverable.
- **If the rubric requires a database:** Implement a thin optional layer (not in the default path):
  - Environment: `ENABLE_SQLITE=1`, `SQLITE_PATH=./data/extractions.db`
  - Table example: `extractions(id TEXT PRIMARY KEY, created_at TEXT, filename TEXT, profile_json TEXT, metadata_json TEXT)`
  - Write **after** successful `POST /extract` (e.g. `BackgroundTasks`) so latency stays predictable.
  - Document in the final report and update [PROPOSAL_TECHNICAL_ALIGNMENT.md](PROPOSAL_TECHNICAL_ALIGNMENT.md) when enabled.

Until then, use the sign-off line below.

## What we ship

- **Stateless API:** `POST /extract` returns JSON immediately. No database is required for the core demo or Definition of Done in the execution plan.
- **Optional cloud storage:** S3 (or similar) can archive PDFs or responses **asynchronously** behind an env flag — not implemented in the default path so local and CI stay simple.
- **SQLite:** Not wired into the FastAPI app. If the course rubric requires “persistence,” treat SQLite as an **optional add-on** (e.g. append-only table: `request_id`, path, timestamp) or align the written proposal with this file.

## Supervisor sign-off line (paste into report if needed)

> “MVP scope: extraction API is stateless; SQLite/S3 are documented as optional. Full persistence is out of scope unless explicitly added.”

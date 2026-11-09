# Research Observatory on GitHub Pages

Public workspace for APEX-G literature and RasaChatbot evaluation. Original research and clinical repositories remain authoritative.

## Run

Python 3: `python -m http.server 8080 --directory static/research-observatory` from the public repository root. Open http://localhost:8080/ . No installation or backend is required for this static version.

The existing Hugo GitHub Actions workflow deploys `static/research-observatory` to the public `/research-observatory/` URL on every main-branch push.

## Data and persistence

The dated APEX-G snapshot contains bibliographic records, provenance, review links and short existing review excerpts. No clinical records or provider credentials are bundled.

Clinical JSON/JSONL imports, per-paper evidence/Markdown drafts, relevance labels and synthesis drafts remain in browser memory. Use Download backup and Restore backup for continuity. Evaluation import replaces loaded runs; restoring a backup replaces current drafts and runs. No records are uploaded by this app. Missing metrics and ungraded claims remain unknown.

Reload snapshot reloads the published JSON. It does not synchronize private repositories. Public Pages cannot run the SQL API or safely hold provider credentials. The optional private SQL backend source is included separately in `research-observatory-source/`; the live private companion remains https://research-observatory-arman.prettyagents-9813.chatgpt.site .

## Architecture

Custom Python orchestration / SQL-first retrieval / SSE -> de-identified evaluation export -> JSON/JSONL import -> timeline, claim audits, descriptive variant metrics, matched-scenario coverage, ablations and stress views. The workspace does not execute clinical scenarios or assert clinical validity.

Full source: `research-observatory-source/`. Static assets and current snapshot: `static/research-observatory/`.

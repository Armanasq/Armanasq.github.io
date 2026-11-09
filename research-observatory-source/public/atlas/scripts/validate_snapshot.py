#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "data/snapshot.json").read_text(encoding="utf-8"))
assert data["schema_version"] == "1.0.0"
assert data["reconciliation"]["counts_consistent"] is True
assert data["data_classification"] == "curated_bibliographic_metadata_and_short_review_excerpts"
assert isinstance(data["papers"], list) and data["papers"]
assert not data["evaluation"]["runs"]
ids = [p["id"] for p in data["papers"]]
assert len(ids) == len(set(ids))
assert all(p["review_url"] is None or p["review_url"].startswith("https://github.com/") for p in data["papers"])
for p in data["papers"]:
    assert p["relevance"] == "unassessed"
    assert p["archive_integrity"] in ("repository_auditor_passed", "not_individually_verified")
    for key in ("research_questions", "methods", "datasets", "claims", "limitations", "contradictions", "open_problems"):
        assert key in p["review_excerpts"]
assert (root / "index.html").is_file()
assert (root / "app.js").is_file()
assert (root / "styles.css").is_file()
print(json.dumps({"result":"pass","records":len(ids),"scientific_gates":dict(Counter(p["reading_status"] for p in data["papers"])),"pdfs_reported":data["summary"]["pdfs_reported"]},indent=2))


#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "Armanasq/APEX-G"
RASA_REPOS = ("Armanasq/rasachatbot-backend", "Armanasq/rasachatbot-ui")
FIELDS = {
    "research_questions": ("research question", "hypothesis", "research objective"),
    "methods": ("method", "mechanism", "experimental design", "causal model", "protocol", "intervention"),
    "datasets": ("dataset", "benchmark", "task family", "data source"),
    "claims": ("finding", "result", "quantitative evidence", "empirical evidence", "evidence", "reported outcome"),
    "limitations": ("limitation", "validity", "confound", "caveat", "reproducib", "verification gate"),
    "contradictions": ("contradiction", "conflict", "negative result", "counterexample", "competing explanation"),
    "open_problems": ("open problem", "future work", "remaining", "outstanding", "gap", "falsifiable experiment"),
}
TOPICS = {
    "active_problem_solving": "Active problem-solving",
    "rl_and_curriculum": "RL and curriculum",
    "transfer_and_generalization": "Transfer and generalization",
    "psychometrics_and_intelligence": "Psychometrics and intelligence",
    "benchmarks_and_evaluation": "Benchmarks and evaluation",
    "human_cognitive_training": "Human cognitive training",
}
BASE = "https://github.com/" + REPO + "/blob/main/"

def rows(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def clean(value):
    if not value:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("none", "null", "no doi verified", "n/a", "unknown") else None

def github_file(path):
    if not path:
        return None
    rel = str(path).lstrip("/")
    if rel.startswith("../") or rel.startswith("/"):
        return None
    return BASE + "/".join(quote(part, safe="") for part in rel.split("/"))

def arxiv_id(*values):
    for val in values:
        if not val:
            continue
        match = re.search(r"(?:arxiv\.|arxiv:|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(v\d+)?", str(val), re.I)
        if match:
            return match.group(1) + (match.group(2) or "")
    return None

def sections_from_review(path):
    if not path.is_file():
        return {}, []
    content = path.read_text(encoding="utf-8", errors="replace")
    headings = []
    sections = defaultdict(list)
    current = None
    for line in content.splitlines():
        if line.startswith("#"):
            heading = re.sub(r"^#{1,6}\s*", "", line).strip()
            if heading:
                headings.append(heading[:140])
                current = heading.casefold()
            continue
        if not current or not line.strip():
            continue
        for field, keys in FIELDS.items():
            if any(key in current for key in keys):
                sections[field].append(line.strip())
    result = {}
    for field in FIELDS:
        if sections.get(field):
            excerpt = "\n".join(sections[field])
            result[field] = excerpt[:850] + ("…" if len(excerpt) > 850 else "")
        else:
            result[field] = None
    return result, headings[:48]

def get_commits(repository, token):
    if not token:
        return {"status": "unavailable", "reason": "No repository-read credential configured", "commits": []}
    req = Request(
        "https://api.github.com/repos/" + repository + "/commits?per_page=12",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
            "User-Agent": "APEX-G-research-workspace",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(req, timeout=15) as response:
            data = json.load(response)
        return {
            "status": "synced",
            "commits": [{
                "sha": row["sha"],
                "message": (row.get("commit") or {}).get("message", "").split("\n")[0][:220],
                "date": ((row.get("commit") or {}).get("author") or {}).get("date"),
                "url": row.get("html_url"),
            } for row in data if isinstance(row, dict) and row.get("sha")],
        }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__, "commits": []}

def build(root, token=None, rasa_token=None):
    lit = root / "literature_review"
    bibliography = rows(lit / "papers.csv")
    inventory = {r["paper_id"]: r for r in rows(lit / "processing_inventory.csv") if r.get("paper_id")}
    manifests = {r["paper_id"]: r for r in rows(lit / "sources/download_manifest.csv") if r.get("paper_id")}
    crosswalk = rows(lit / "sources/source_id_crosswalk.csv")
    integrity = read_json(lit / "sources/corpus_integrity_report.json")
    manual = rows(lit / "sources/manual_acquisition_required.csv")
    restricted = rows(lit / "sources/restricted_or_link_only.csv")
    inverse = {}
    for row in crosswalk:
        for alias in [row.get("acquisition_id", ""), row.get("inventory_id", "")] + row.get("index_ids", "").split(";"):
            if alias:
                inverse[alias.strip()] = row
    grouped = {}
    for bib in bibliography:
        paper_id = clean(bib.get("paper_id"))
        if not paper_id:
            continue
        cross = inverse.get(paper_id, {})
        canonical = clean(cross.get("acquisition_id")) or paper_id
        if canonical not in grouped:
            grouped[canonical] = {"record_id": canonical, "index_ids": [], "bibliography": []}
        grouped[canonical]["index_ids"].append(paper_id)
        grouped[canonical]["bibliography"].append(bib)
    for paper_id, inv in inventory.items():
        cross = inverse.get(paper_id, {})
        canonical = clean(cross.get("acquisition_id")) or paper_id
        if canonical not in grouped:
            grouped[canonical] = {"record_id": canonical, "index_ids": [], "bibliography": []}
    records = []
    for canonical, group in grouped.items():
        bibs = group["bibliography"]
        cross = inverse.get(canonical, {})
        inv_id = clean(cross.get("inventory_id")) or canonical
        inv = inventory.get(inv_id) or inventory.get(canonical) or {}
        manifest = manifests.get(canonical) or manifests.get(inv_id) or {}
        preferred = sorted(bibs, key=lambda b: (
            b.get("access_status", "") not in ("downloaded", "read_complete_analysis_committed"),
            not bool(clean(b.get("landing_url"))),
            not bool(clean(b.get("publication_doi"))),
        ))
        bib = preferred[0] if preferred else {}
        title = clean(bib.get("title")) or clean(manifest.get("title")) or canonical.replace("_", " ")
        top = clean(inv.get("topic")) or clean(bib.get("primary_topic"))
        if top:
            top = re.sub(r"^\d\d_", "", top)
        year_str = clean(inv.get("year")) or clean(bib.get("year")) or clean(manifest.get("year"))
        try:
            year = int(year_str) if year_str else None
        except ValueError:
            year = None
        source_path = clean(manifest.get("path")) or clean(inv.get("source_pdf")) or clean(bib.get("full_text_file"))
        analysis_path = clean(inv.get("analysis_file"))
        review_exists = bool(analysis_path and (root / analysis_path).is_file())
        excerpts, headings = sections_from_review(root / analysis_path) if review_exists else ({k: None for k in FIELDS}, [])
        pub_doi = clean(bib.get("publication_doi"))
        pre_doi = clean(bib.get("preprint_doi"))
        declared_doi = pub_doi or pre_doi or clean(manifest.get("doi"))
        aid = arxiv_id(pre_doi, declared_doi, bib.get("landing_url"), manifest.get("source_url"))
        has_archive = bool(manifest and clean(manifest.get("path")))
        reading = clean(inv.get("reading_status")) or "not_recorded"
        access = clean(bib.get("access_status")) or ("archived" if has_archive else "not_recorded")
        download_url = clean(manifest.get("source_url"))
        note = clean(inv.get("notes"))
        flags = []
        if not has_archive and access not in ("link_only_license_not_verified",):
            flags.append("acquisition_unverified")
        if access == "link_only_license_not_verified":
            flags.append("license_or_access_restricted")
        if not review_exists:
            flags.append("review_missing_or_unverified")
        if reading == "text_complete_visual_pending":
            flags.append("native_visual_verification_pending")
        if reading == "critical_screening":
            flags.append("primary_source_review_pending")
        if reading == "not_recorded":
            flags.append("scientific_status_unknown")
        if clean(inv.get("metadata_status")) == "metadata_pending":
            flags.append("bibliographic_metadata_pending")
        if not declared_doi:
            flags.append("doi_unverified")
        records.append({
            "id": canonical,
            "paper_id": inv_id,
            "index_ids": sorted(set(group["index_ids"])),
            "title": title,
            "year": year,
            "venue": clean(bib.get("venue")),
            "topic": top or "unclassified",
            "topic_label": TOPICS.get(top, top.replace("_", " ").title() if top else "Unclassified"),
            "evidence_type": clean(bib.get("evidence_type")),
            "relevance": "unassessed",
            "publication_doi": pub_doi,
            "preprint_doi": pre_doi,
            "doi": declared_doi,
            "doi_url": "https://doi.org/" + quote(declared_doi, safe="/.") if declared_doi else None,
            "arxiv_id": aid,
            "arxiv_url": "https://arxiv.org/abs/" + aid if aid else None,
            "publisher_url": clean(bib.get("landing_url")),
            "source_url": download_url,
            "pdf_url": github_file(source_path) if has_archive else None,
            "full_text_path": source_path if has_archive else None,
            "review_url": github_file(analysis_path) if review_exists else None,
            "review_path": analysis_path if review_exists else None,
            "reading_status": reading,
            "extraction_status": clean(inv.get("extraction_status")) or "not_recorded",
            "metadata_status": clean(inv.get("metadata_status")) or "not_recorded",
            "acquisition_status": access,
            "archive_integrity": "repository_auditor_passed" if has_archive and integrity.get("summary", {}).get("errors") == 0 and integrity.get("summary", {}).get("warnings") == 0 else "not_individually_verified",
            "license": clean(manifest.get("license")) or clean(bib.get("license")),
            "source_sha256": clean(manifest.get("sha256")),
            "review_excerpts": excerpts,
            "review_headings": headings,
            "review_provenance": github_file(analysis_path) if review_exists else None,
            "inventory_note": (note[:600] + "…" if note and len(note) > 600 else note),
            "flags": flags,
        })
    records.sort(key=lambda p: (-p["year"] if p["year"] else 0, p["title"].casefold()))
    archive_records = [p for p in records if p["full_text_path"]]
    by_status = Counter(inv.get("reading_status") or "not_recorded" for inv in inventory.values())
    title_groups = defaultdict(list)
    for p in records:
        title_groups[" ".join(p["title"].casefold().split())].append(p["id"])
    duplicate_candidates = [ids for ids in title_groups.values() if len(ids) > 1]
    summary = integrity.get("summary", {})
    checks = {
        "inventory_pdf_count": len([i for i in inventory.values() if str(i.get("source_pdf", "")).lower().endswith(".pdf")]),
        "manifest_records": len(manifests),
        "manifest_pdf_count": sum(1 for m in manifests.values() if str(m.get("format", "")).casefold() in ("pdf", "pmc_oa_pdf")),
        "integrity_report_pdf_count": summary.get("pdf_files_on_disk"),
        "integrity_errors": summary.get("errors"),
        "integrity_warnings": summary.get("warnings"),
    }
    checks["counts_consistent"] = (
        checks["inventory_pdf_count"] == checks["manifest_pdf_count"] ==
        checks["integrity_report_pdf_count"]
    ) if isinstance(checks["integrity_report_pdf_count"], int) else None
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_classification": "curated_bibliographic_metadata_and_short_review_excerpts",
        "source_repo": REPO,
        "scope": "Repository inventory and author-authored review excerpts; not independent verification",
        "summary": {
            "bibliography_rows": len(bibliography),
            "canonical_records": len(records),
            "acquired_sources_reported": summary.get("manifest_files_present"),
            "pdfs_reported": summary.get("pdf_files_on_disk"),
            "pdfs_in_inventory": checks["inventory_pdf_count"],
            "reading_statuses": dict(by_status),
            "duplicate_title_candidates": len(duplicate_candidates),
            "integrity_errors": summary.get("errors"),
            "integrity_warnings": summary.get("warnings"),
        },
        "reconciliation": checks,
        "papers": records,
        "acquisition_queue": [{
            "paper_id": r.get("paper_id"),
            "title": r.get("title"),
            "doi": clean(r.get("doi")),
            "url": clean(r.get("official_url")),
            "status": r.get("acquisition_status"),
            "reason": r.get("reason"),
        } for r in manual],
        "restricted_links": [{
            "paper_id": r.get("paper_id"),
            "title": r.get("title"),
            "url": r.get("source_url"),
            "reason": r.get("reason"),
        } for r in restricted],
        "duplicate_candidates": duplicate_candidates,
        "synthesis": {
            "research_framework": github_file("literature_review/research_framework.md"),
            "comparative_analysis": github_file("literature_review/comparative_analysis.md"),
            "gaps": github_file("literature_review/research_gaps.md"),
            "working_review": github_file("literature_review/synthesis/working_literature_review.md"),
            "missing_literature": github_file("literature_review/synthesis/missing_literature_audit.md"),
            "qa_moderator": github_file("literature_review/qa_moderator_audit.md"),
            "references": github_file("literature_review/references.md"),
            "per_paper_audit": github_file("literature_review/synthesis/per_paper_review_verification_2026-11-09.md"),
        },
        "research_question": "Does repeated active problem-solving, combined with feedback and parameter updates, improve persistent transfer to structurally novel cognitive domains beyond exposure and inference-compute effects?",
        "evaluation": {"source": "No clinical evaluation records are published from private sources", "runs": [], "status": "awaiting_explicit_local_import"},
        "repositories": {
            REPO: {
                **get_commits(REPO, token),
                "url": "https://github.com/" + REPO,
                "branch": "main",
                "build_commit": os.environ.get("GITHUB_SHA") or None,
            },
            **{
                repo: {
                    **get_commits(repo, rasa_token),
                    "url": "https://github.com/" + repo,
                    "branch": "main",
                }
                for repo in RASA_REPOS
            },
        },
        "provenance": [
            {"path": "literature_review/papers.csv", "kind": "bibliography"},
            {"path": "literature_review/processing_inventory.csv", "kind": "scientific_reading_ledger"},
            {"path": "literature_review/sources/download_manifest.csv", "kind": "source_acquisition"},
            {"path": "literature_review/sources/source_id_crosswalk.csv", "kind": "alias_resolution"},
            {"path": "literature_review/sources/corpus_integrity_report.json", "kind": "integrity_audit"},
            {"path": "literature_review/sources/manual_acquisition_required.csv", "kind": "acquisition_exceptions"},
            {"path": "literature_review/sources/restricted_or_link_only.csv", "kind": "licensing_restrictions"},
        ],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "research-workspace/data/snapshot.json").resolve()
    if not (root / "literature_review/papers.csv").is_file():
        parser.error("Root must contain literature_review/papers.csv")
    snapshot = build(root, os.environ.get("GITHUB_TOKEN"), os.environ.get("RESEARCH_READ_TOKEN"))
    if snapshot["reconciliation"]["counts_consistent"] is False:
        raise SystemExit("Integrity mismatch between inventory, manifest and report; refusing a misleading snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "records": len(snapshot["papers"]),
        "reading": snapshot["summary"]["reading_statuses"],
        "pdfs": snapshot["summary"]["pdfs_reported"],
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()


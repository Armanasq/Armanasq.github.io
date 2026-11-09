# Research Atlas — APEX-G + RasaChatbot

This is a separate, read-only research workspace served from GitHub Pages. It does not replace or modify the existing literature review, clinical Python backend, or Next.js clinical chat UI.

## Pages deployment

The workspace is deployed at **https://armanasq.github.io/research-atlas/** from the existing `Armanasq/Armanasq.github.io` GitHub Pages site. The Hugo homepage and both RasaChatbot repositories remain unchanged. Deployment evidence: https://github.com/Armanasq/Armanasq.github.io/actions/runs/42582350986.

The APEX-G repository remains the authoritative source for the frontend and literature corpus. Its Actions workflow compiles, validates and commits a sanitized snapshot to APEX-G main on source changes and scheduled runs. An independent Pages site for private APEX-G is not currently enabled; that job is skipped rather than failing.

**Refresh boundary:** The public personal-site copy is a point-in-time mirror, not a live connection to private APEX-G. To enable automatic cross-repository publication, add repository secret `RESEARCH_ATLAS_PUBLIC_PUSH_TOKEN` to APEX-G. It must be a fine-grained GitHub personal access token with `Contents: Read and write` for `Armanasq/Armanasq.github.io` only. When configured, the source workflow checks out that public repository, copies the validated frontend and snapshot, and commits to `main` without touching the Hugo site's other files. The public Hugo workflow then deploys that change.

## Connected authoritative sources

- APEX-G papers.csv: bibliographic index with DOI, preprint DOI, venue, topic, year, publication links.
- processing_inventory.csv: scientific reading statuses, extraction statuses, review Markdown paths.
- download_manifest.csv: source URL, PDF path, checksum, license and acquisition status.
- source_id_crosswalk.csv: alias reconciliation.
- corpus_integrity_report.json: repository-reported acquisition audit.
- manual_acquisition_required.csv and restricted_or_link_only.csv: distinct acquisition/permission queues.
- Existing literature_review/papers/*.md: one primary source review per study where recorded.
- Existing synthesis research framework, comparative analyses, reference list, QA audit and working final review.
- RasaChatbot backend and UI repositories: linked, not embedded.

## Data and application architecture

APEX-G main -> Python standard-library snapshot compiler -> validation -> sanitized JSON -> GitHub Pages single-page application.

RasaChatbot clinical Python orchestration / SQL-first retrieval / streaming -> de-identified evaluation adapter and export -> manually import JSON or JSONL into browser memory -> scenario timeline, claim audit, variant comparisons, ablations, stress tests, metrics and exports.

The static Pages application cannot securely connect to private runtime databases or push to GitHub from the browser. Clinical evaluation results are not automatically assumed to exist. Missing fields stay null, marked unknown, or unassessed. The UI shows only source-authored short Markdown excerpts, never generated scientific claims.

## Local development

From the APEX-G checkout root, run:

    python research-workspace/scripts/build_snapshot.py
    python research-workspace/scripts/validate_snapshot.py
    python -m http.server 8080 --directory research-workspace

Open http://localhost:8080/ .

No npm packages or JavaScript frameworks are required. The compiler depends only on Python 3.10+ standard library.

The compiler fails when the source audit and manifest disagree on the PDF archive count. This is intentional: count conflicts must not silently be presented as verified.

## Clinical evaluation import

Open **Scenario evaluation → Download JSON template** to obtain the input structure. The full contract is in EVALUATION_SCHEMA.md. Export de-identified production or test data using a Python adapter that maps real evaluator telemetry into the contract.

Required: run_id, scenario_id, variant, turns. Optional measured per-run values: latency_ms, total_tokens, cost_usd, outcome, validation_errors, contradictions, clinical_review_status, stress_category, ablations. Optional per-turn values: workflow, decision, selected tool, state_before, state_delta, clarification/referral, retrieval backend, evidence provenance, claim-level support, clinical review flags, validation state.

The UI can import/export runs, inspect a turn timeline, identify unsupported or contradicted claims, and compare descriptive group totals. It cannot determine citation validity or clinical correctness on its own. Run variants must be evaluated on matching scenarios and measured under comparable conditions.

## Synchronization

- Changes to APEX-G literature files or research-workspace code on main trigger a Pages rebuild; there is also a scheduled daily build and manual dispatch.
- The workflow collects APEX-G commit metadata using its GitHub Actions token.
- Optional repository secret RESEARCH_READ_TOKEN, scoped to **read-only** access to the other two private RasaChatbot repositories, enables their commit feeds. Missing access is explicitly shown as unavailable.
- The site never performs Git mutations or replaces source-controlled READMEs. README links provide navigation and provenance to the canonical living documents.
- Actual deployment must be inspected under Actions → Research Atlas Pages. Do not infer completion from committed workflow configuration.

## Privacy boundary

Publishing Pages from a private repo can expose static artifacts publicly. The compiled snapshot includes bibliographic metadata and short portions of source-authored reviews, but no complete PDFs, protected clinical runs, credentials or backend configuration. Review that public exposure before enabling Pages. Never import identifiable patient records into a public-browser session. The tool does not use localStorage or send imported evaluation content to a backend.

## Uncertainty semantics

The paper relevance field is unassessed unless independently curated. A missing mapped review section is not proof that a paper omits that dimension. Archive integrity is not scientific verification; text_complete_visual_pending is not read_complete; read_complete is a repository-authored status, not a fresh independent replication. BACIE/COTAR scores are unavailable until real runs are imported.


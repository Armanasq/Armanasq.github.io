# Evaluation telemetry contract v1

The GitHub Pages app imports de-identified JSON, a JSON array, or newline-delimited JSON. Import remains in browser memory until explicitly saved to the private SQL workspace. It is never uploaded to GitHub by this application. It is separate from the current production RasaChatbot UI and streaming backend.

## Required run fields
- run_id: nonempty, unique string
- scenario_id: nonempty string used to align variants
- variant: baseline, BACIE, COTAR, or explicitly named intervention
- turns: ordered array of turn objects

## Optional run fields
- outcome: string or null
- latency_ms, total_tokens, cost_usd: nonnegative measured number or null
- clinical_review_status: approved, flagged, unreviewed, or null
- validation_errors, contradictions, ablations: arrays of strings
- stress_category: string or null

## Per-turn fields
- role, message: de-identified text
- workflow, decision, tool: recorded router and tool decisions, never heuristically reconstructed
- state_before, state_delta: structured objects or null
- clarification_or_referral, retrieval_backend, validation_status: recorded values or null
- provenance: array of objects containing source_id, chunk_id and/or url
- claims: array of objects with text, support_status, evidence_ids (string array), clinical_review_flag

An empty claims array means not audited, **not zero unsupported claims**. A missing value must remain null; no synthetic scores.

## Integration contract
Use a custom Python adapter at the existing decision evaluator and runtime event boundary. Collect actual SQL/FTS retrieval provenance, streaming timings, state deltas, tool decisions, validation failures, and ledger metering. Export a de-identified file only after explicit clinical-data review. Do not treat examples as observed runs.

Comparisons require matched scenario IDs, equivalent conditions and controlled intervention definitions. Descriptive group totals and latency quantiles are not causal performance claims. Clinical-review flags require qualified review.

GitHub Pages cannot securely host backend credentials or protected patient data. Never embed private GitHub or clinical API tokens in frontend JavaScript.


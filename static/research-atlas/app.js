"use strict";
const state = { snapshot: null, page: "overview", paper: null, query: "", topic: "all", year: "all", status: "all", evidenceField: "all", evalRuns: [], selectedRun: null, scenarioFilter: "all", runImportLabel: null };
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value === null || value === undefined ? "" : value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (v) => v === null || v === undefined || v === "" ? "Unknown" : esc(v);
const num = (v) => typeof v === "number" && Number.isFinite(v) ? new Intl.NumberFormat("en-US").format(v) : "Unknown";
const link = (url, label, cls = "linkchip") => url && /^https:\/\//i.test(url) ? '<a class="' + cls + '" target="_blank" rel="noopener noreferrer" href="' + esc(url) + '">' + esc(label) + ' ↗</a>' : "";
const badge = (name) => {
  const text = String(name || "unknown").replaceAll("_", " ");
  const mood = /read.complete|verified|supported|passed|success|baseline/i.test(String(name)) ? "good" : /error|fail|contradict|missing|unsupported|critical/i.test(String(name)) ? "bad" : /pending|unknown|unassessed|unavailable|clarif|referral|manual|warning/i.test(String(name)) ? "warn" : "";
  return '<span class="tag ' + mood + '">' + esc(text) + "</span>";
};
const stat = (label, value, foot) => '<div class="stat"><div class="stat-label">' + esc(label) + '</div><div class="stat-value' + (value === null || value === undefined ? ' unknown' : '') + '">' + num(value) + '</div><div class="stat-foot">' + esc(foot || "") + "</div></div>";
const head = (eyebrow, title, subtitle, actions = "") => '<div class="heading"><div><div class="eyebrow">' + esc(eyebrow) + '</div><h1>' + esc(title) + '</h1><p class="subtitle">' + esc(subtitle) + '</p></div>' + actions + "</div>";
const sec = (title, sub = "", right = "") => '<div class="section-head"><div><h2>' + esc(title) + '</h2>' + (sub ? '<p>' + esc(sub) + '</p>' : "") + "</div>" + right + "</div>";
const empty = (title, body) => '<div class="empty"><strong>' + esc(title) + '</strong><p>' + esc(body) + "</p></div>";
const entry = (label, value) => '<div class="kv"><b>' + esc(label) + '</b><span>' + fmt(value) + '</span></div>';
const htmlDownload = (filename, content, mime) => {
  const blob = new Blob([content], {type: mime});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
};
const csv = (rows, fields) => [fields.join(","), ...rows.map(row => fields.map(k => '"' + String(row[k] ?? "").replaceAll('"', '""') + '"').join(","))].join("\r\n");
const notify = (message) => {const node = $("#toast"); node.textContent = message; node.classList.add("visible"); setTimeout(() => node.classList.remove("visible"), 3200);};
const safeString = (v) => typeof v === "string" ? v : v == null ? null : String(v);
const countBy = (rows, key) => rows.reduce((out, row) => { const name = row[key] || "Unclassified"; out[name] = (out[name] || 0) + 1; return out; }, {});
const bars = (items, mode) => {const max = Math.max(1, ...items.map(item => item[1]));return items.map(([key, amount]) => '<div class="barrow"><span class="barlabel" title="' + esc(key) + '">' + esc(key.replaceAll("_", " ")) + '</span><span class="bartrack"><span class="barfill ' + (mode || "") + '" style="display:block;width:' + Math.max(0, Math.min(100, amount / max * 100)) + '%"></span></span><span class="barcount">' + num(amount) + '</span></div>').join("") || empty("No data", "No verified counts are available.");};
const papers = () => state.snapshot?.papers || [];
const years = () => [...new Set(papers().map(p => p.year).filter(y => typeof y === "number"))].sort((a,b)=>b-a);
const topics = () => [...new Set(papers().map(p => p.topic))].sort();
const filteredPapers = () => papers().filter(p => {
  const q = state.query.toLowerCase();
  return (state.topic === "all" || p.topic === state.topic) &&
    (state.year === "all" || String(p.year) === state.year) &&
    (state.status === "all" || (state.status === "missing" ? p.flags.some(f=>/missing|unverified|pending|unknown/.test(f)) : p.reading_status === state.status)) &&
    (!q || [p.title,p.id,p.venue,p.publication_doi,p.preprint_doi,p.topic_label,p.evidence_type].some(v => String(v || "").toLowerCase().includes(q)));
});
const select = (id, value, options) => '<select class="control" id="' + id + '">' + options.map(([v,t])=>'<option value="' + esc(v) + '"' + (String(value) === String(v) ? ' selected' : '') + '>' + esc(t) + '</option>').join("") + '</select>';
const paperFilters = () => '<div class="toolbar"><label class="sr-only" for="paper-search">Search papers</label><input class="control search" id="paper-search" type="search" placeholder="Search title, DOI, ID, methods…" value="' + esc(state.query) + '" aria-label="Search papers">' +
  select("topic-filter",state.topic,[["all","All topics"],...topics().map(x=>[x,x.replaceAll("_"," ")])]) +
  select("year-filter",state.year,[["all","All years"],...years().map(x=>[x,x])]) +
  select("status-filter",state.status,[["all","All review states"],["read_complete","Read complete"],["text_complete_visual_pending","Visual verification pending"],["critical_screening","Critical screening"],["not_recorded","Not recorded"],["missing","Requires attention"]]) +
  '<button class="button outline small" id="export-papers">Export CSV</button></div>';
const paperTable = (rows) => '<div class="table-scroll"><table><thead><tr><th>Paper / study</th><th>Year</th><th>Topic</th><th>Scientific review gate</th><th>Archive</th><th>Record</th></tr></thead><tbody>' +
  rows.slice(0,350).map(p => '<tr class="selectable" data-paper="' + esc(p.id) + '" tabindex="0" role="button" aria-label="Inspect ' + esc(p.title) + '"><td class="title-cell">' + esc(p.title) + '<div class="fine">' + esc(p.id) + '</div></td><td>' + fmt(p.year) + '</td><td class="muted">' + esc(p.topic_label) + '</td><td>' + badge(p.reading_status) + '</td><td>' + badge(p.full_text_path ? "archived" : p.acquisition_status) + '</td><td>' + (p.review_url ? '<span class="tag good">Markdown linked</span>' : badge("not recorded")) + "</td></tr>").join("") + "</tbody></table></div>";
const renderOverview = () => {
  const s = state.snapshot.summary;
  const statuses = s.reading_statuses || {};
  const topicCounts = Object.entries(countBy(papers(), "topic")).sort((a,b)=>b[1]-a[1]);
  const yearCounts = Object.entries(countBy(papers(), "year")).sort((a,b)=>Number(a[0])-Number(b[0]));
  const outstanding = papers().filter(p=>p.reading_status === "text_complete_visual_pending").slice(0,6);
  return head("Research atlas / overview", "The state of the research.", "A working view of the APEX-G evidence corpus and RasaChatbot evaluation. Figures come from the latest published source snapshot, not fixed dashboard constants.") +
    '<div class="editorial-summary"><div><span class="editorial-label">RESEARCH NOTEBOOK / LIVING CORPUS</span><p class="editorial-intro">Follow the papers. Inspect the evidence. Track what remains unresolved.</p></div><div class="editorial-meta"><strong>'+num(papers().length)+'</strong><span>canonical research records</span><small>Last compiled '+esc(isoTime(state.snapshot.generated_at))+'</small></div></div>' +
    '<div class="stats">' + stat("Archived PDFs",s.pdfs_reported,"Repository integrity report") + stat("Read complete",statuses.read_complete ?? null,"Scientific review inventory") + stat("Visual verification pending",statuses.text_complete_visual_pending ?? null,"Not final scientific certification") + stat("Imported clinical runs",state.evalRuns.length,"Browser session only") + "</div>" +
    '<div class="section"><div class="two-col"><div class="panel">' + sec("Research coverage","Canonical, alias-reconciled bibliographic records") + bars(topicCounts.slice(0,9)) + '</div><div class="panel">' + sec("Publication years","Indexed bibliographic records") + bars(yearCounts,"teal") + '</div></div></div>' +
    '<div class="section">' + sec("Verification queue","Original-text review is not equivalent to native figure verification",'<button class="button secondary small" data-go="papers" data-status="missing">Open full queue →</button>') + '<div class="panel tight">' + paperTable(outstanding) + '</div></div>' +
    '<div class="section">' + sec("Connected projects","Separate codebases; common observability layer") + '<div class="three-col">' +
    projectCard("APEX-G","Primary research corpus, manifests, reviews, synthesis","https://github.com/Armanasq/APEX-G") +
    projectCard("RasaChatbot backend","Python orchestration, clinical decisions, retrieval, SSE","https://github.com/Armanasq/rasachatbot-backend") +
    projectCard("RasaChatbot UI","Existing production frontend; remains unchanged","https://github.com/Armanasq/rasachatbot-ui") +
    '</div></div>' +
    '<div class="section"><div class="note warn-note">Publication safety: this static site contains only the generated bibliographic snapshot and short, source-authored research excerpts. Private repository access and clinical evaluation records are not granted through GitHub Pages. All absent measurements are shown as unavailable.</div></div>';
};
const projectCard = (name,description,url) => '<div class="panel"><h3>' + esc(name) + '</h3><p class="muted" style="min-height:43px;font-size:12px;line-height:1.6">' + esc(description) + '</p>' + link(url,"Open repository") + "</div>";
const renderPapers = () => {
  const rows = filteredPapers();
  return head("APEX-G / acquisition registry","Paper acquisition & verification","Canonical study identities are reconciled against source ID aliases, source manifests and the scientific reading inventory. Select a row to inspect provenance and the existing Markdown review.",
    '<button class="button secondary" id="export-papers-top">Export filtered records</button>') +
    '<div class="stats">' + stat("Canonical studies",papers().length,"From current bibliography + inventory") + stat("Records visible",rows.length,"After filters") + stat("Manual download queue",state.snapshot.acquisition_queue.length,"Acquisition exceptions") + stat("Duplicate-title candidates",state.snapshot.summary.duplicate_title_candidates,"Needs review, not automatically merged") + '</div>' +
    '<div class="section">' + sec("Bibliographic index","Topic, year, review status, verified source links") +
    '<div class="panel tight">' + paperFilters() + paperTable(rows) + '<div class="table-meta"><span>' + num(rows.length) + ' matching canonical records</span><span>Source: papers.csv · processing_inventory.csv · crosswalk.csv</span></div></div></div>' +
    '<div class="section">' + sec("Manual acquisition required","These are recorded exceptions, not claims that the paper is absent from the literature") +
    '<div class="panel tight"><div class="table-scroll"><table><thead><tr><th>Paper</th><th>Status</th><th>Reason</th><th>Source</th></tr></thead><tbody>' +
    state.snapshot.acquisition_queue.map(p=>'<tr><td class="title-cell">' + esc(p.title || p.paper_id) + '</td><td>' + badge(p.status) + '</td><td class="muted">' + esc(p.reason || "Unspecified") + '</td><td>' + link(p.url,"Publisher") + '</td></tr>').join("") + '</tbody></table></div></div></div>';
};
const renderPaperDetail = () => {
  const p = papers().find(p=>p.id === state.paper);
  if (!p) {state.paper = null;return renderPapers();}
  const blocks = Object.entries(p.review_excerpts || {});
  return '<button class="button secondary small" id="back-papers">← All papers</button><div style="height:16px"></div>' +
    head("Research evidence / " + p.id,p.title,"One evidence record per canonical paper, linked to the author-maintained Markdown review. Excerpts reflect named review sections, not machine-verified source claims.") +
    '<div class="pill-row">' + badge(p.reading_status) + badge(p.metadata_status) + badge(p.acquisition_status) + badge(p.relevance) + '</div>' +
    '<div class="detail"><div><div class="panel"><h3>Evidence and methodological appraisal</h3>' +
    '<div class="evidence-grid">' + blocks.map(([key,value])=>'<div class="evidence-card"><h3>' + esc(key.replaceAll("_"," ")) + '</h3>' + (value ? '<div class="excerpt">' + esc(value) + '</div>' : '<p class="fine">Not independently structured in source record. Consult the full Markdown review.</p>') + '</div>').join("") + '</div></div>' +
    '<div class="section"><div class="panel"><h3>Verification ledger</h3><div class="detail-grid">' +
    entry("Reading gate",p.reading_status) + entry("Text extraction",p.extraction_status) + entry("Metadata gate",p.metadata_status) + entry("Source archive integrity",p.archive_integrity) + entry("Relevance classification",p.relevance) + entry("Primary evidence classification",p.evidence_type) + '</div><p class="fine">' + esc(p.inventory_note || "No additional inventory note") + '</p><div class="pill-row">' + p.flags.map(badge).join("") + '</div></div></div></div>' +
    '<div><div class="panel"><h3>Source provenance</h3><div class="detail-grid" style="grid-template-columns:1fr">' +
    entry("Canonical ID",p.id) + entry("Index aliases",p.index_ids.join(", ") || null) + entry("Venue",p.venue) + entry("Year",p.year) + entry("Topic",p.topic_label) + entry("Publication DOI",p.publication_doi) + entry("Preprint DOI",p.preprint_doi) + entry("arXiv ID",p.arxiv_id) + entry("Source SHA-256",p.source_sha256) + entry("Source license",p.license) + '</div><div class="pill-row">' + link(p.publisher_url,"Publisher") + link(p.doi_url,"DOI") + link(p.arxiv_url,"arXiv") + link(p.source_url,"Source") + link(p.pdf_url,"Archived full text") + link(p.review_url,"Markdown review") + '</div></div></div></div>';
};
const renderEvidence = () => {
  const fields = ["research_questions","methods","datasets","claims","limitations","contradictions","open_problems"];
  const matched = filteredPapers();
  const presentCount = fields.map(f=>[f,papers().filter(p=>Boolean(p.review_excerpts?.[f])).length]);
  const withField = matched.filter(p=>state.evidenceField==="all" || Boolean(p.review_excerpts?.[state.evidenceField]));
  return head("APEX-G / evidence explorer","Evidence, methods & limitations","Structured excerpts are extracted conservatively from existing Markdown review headings. A missing excerpt means the evidence field was not automatically mapped, not that the paper lacks the information.") +
    '<div class="stats">' + stat("Linked reviews",papers().filter(p=>p.review_url).length,"Existing Markdown analyses") + stat("Mapped method excerpts",presentCount.find(r=>r[0]==="methods")[1],"Heading-based extraction") + stat("Mapped limitation excerpts",presentCount.find(r=>r[0]==="limitations")[1],"Not independent fact checking") + stat("Mapped contradiction excerpts",presentCount.find(r=>r[0]==="contradictions")[1],"Section-level references only") + '</div>' +
    '<div class="section"><div class="two-col"><div class="panel">' + sec("Evidence field coverage","Review sections mapped by heading") + bars(presentCount) + '</div><div class="panel">' + sec("Review provenance & verification","Independent source interpretation remains mandatory") + '<p class="fine">Each result links to the individual Markdown analysis, original publication, and archived full text where available. Extraction does not infer datasets, statistics or scientific consensus from a title alone.</p><div class="note">Claim-level review requires page or table anchors and the exact original source. Flag unconfirmed figures, conflicting numerical claims, unsupported causal inference and missing independent replication in the source Markdown record.</div></div></div></div>' +
    '<div class="section">' + sec("Explore evidence records","Filter by which structured review field is present") + '<div class="panel tight"><div class="toolbar"><input id="paper-search" class="control search" aria-label="Search papers" type="search" placeholder="Search evidence source…" value="' + esc(state.query) + '">' +
    select("evidence-field",state.evidenceField,[["all","All evidence fields"],...fields.map(f=>[f,f.replaceAll("_"," ")])]) +
    select("topic-filter",state.topic,[["all","All topics"],...topics().map(x=>[x,x.replaceAll("_"," ")])]) +
    '</div>' + paperTable(withField) + '<div class="table-meta"><span>' + withField.length + ' records match</span><button class="button outline small" id="export-papers">Export records</button></div></div></div>';
};
const renderSynthesis = () => {
  const counts = countBy(papers(),"topic_label");
  const assets = state.snapshot.synthesis || {};
  const labels = {
    research_framework:"Research framework & causal estimands",
    comparative_analysis:"Cross-paper comparative analysis",
    gaps:"Research gaps & novelty assessment",
    working_review:"Working final literature review",
    missing_literature:"Missing-literature audit",
    qa_moderator:"Adversarial reviewer audit",
    references:"Working bibliography / references",
    per_paper_audit:"Per-paper verification ledger",
  };
  return head("APEX-G / scholarly synthesis","Research synthesis & open questions","Trace arguments from research questions through study-level evidence to cross-paper comparisons and the working review. Existing synthesis files remain the canonical editable artifacts.") +
    '<div class="panel"><div class="eyebrow">Controlling research question</div><h2 style="font-size:19px;line-height:1.5;font-weight:670;max-width:950px">' + esc(state.snapshot.research_question) + '</h2><div class="note warn-note">The central claim of broadly transferable, persistent latent intelligence growth remains an open empirical question. Benchmark improvement alone is not causal evidence of general intelligence.</div></div>' +
    '<div class="section">' + sec("Evidence-to-synthesis graph","Source files form the audit trail; selection opens upstream records") +
    '<div class="three-col"><div class="panel"><div class="eyebrow">01 · INPUT</div><h3>Primary literature</h3><p class="fine">' + num(papers().length) + ' canonical records · ' + num(papers().filter(p=>p.review_url).length) + ' linked reviews</p><button class="button secondary small" data-go="papers">Inspect paper registry</button></div>' +
    '<div class="panel"><div class="eyebrow">02 · EVIDENCE</div><h3>Methods, claims & counterevidence</h3><p class="fine">Source-linked excerpts, outstanding figure review, competing explanations and limitations.</p><button class="button secondary small" data-go="evidence">Inspect evidence</button></div>' +
    '<div class="panel"><div class="eyebrow">03 · OUTPUT</div><h3>Working scholarly synthesis</h3><p class="fine">Argument synthesis and research gaps, without auto-promoting tentative findings.</p>' + link(assets.working_review,"Open review") + '</div></div></div>' +
    '<div class="section"><div class="two-col"><div class="panel">' + sec("Topic map","Canonical paper records") + bars(Object.entries(counts).sort((a,b)=>b[1]-a[1])) + '</div><div class="panel">' + sec("Canonical working documents","Linked Markdown research assets") +
    '<div class="summary-links">' + Object.entries(labels).map(([key,name])=>link(assets[key],name,"summary-link")).join("") + '</div></div></div></div>' +
    '<div class="section"><div class="panel">' + sec("Missing-paper detection strategy","Two distinct failure modes are preserved") +
    '<div class="two-col"><div><h3>Acquisition gap</h3><p class="fine">A registered paper has no validated downloadable source, requires manual publisher access or has licensing restrictions. Do not treat a link-only source as a failed scientific review.</p><p>' + link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/sources/manual_acquisition_required.csv","Manual queue") + link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/sources/restricted_or_link_only.csv","License restrictions") + '</p></div><div><h3>Coverage gap</h3><p class="fine">A research construct, comparison, publication or negative-control family is absent from the review scope. Candidate detection depends on external bibliographic discovery and independent reconciliation.</p><p>' + link(assets.missing_literature,"Missing-literature audit") + link(assets.gaps,"Research gaps") + '</p></div></div></div></div>';
};
const evalNumeric = (v) => typeof v === "number" && Number.isFinite(v) ? v : null;
const runMetrics = (runs) => {
  const ms = runs.map(r=>evalNumeric(r.latency_ms)).filter(x=>x!==null);
  const p50 = ms.length ? [...ms].sort((a,b)=>a-b)[Math.floor((ms.length-1)*.5)] : null;
  const p95 = ms.length ? [...ms].sort((a,b)=>a-b)[Math.ceil((ms.length-1)*.95)] : null;
  const tokens = runs.map(r=>evalNumeric(r.total_tokens)).filter(x=>x!==null);
  const costs = runs.map(r=>evalNumeric(r.cost_usd)).filter(x=>x!==null);
  const claims = runs.flatMap(r=>r.turns || []).flatMap(t=>t.claims || []);
  const supported = claims.filter(c=>c.support_status === "supported").length;
  return {p50,p95,tokens:tokens.length?tokens.reduce((a,b)=>a+b,0):null,cost:costs.length?costs.reduce((a,b)=>a+b,0):null,
    claims:claims.length, supported, unsupported:claims.filter(c=>c.support_status==="unsupported").length,
    contradicted:claims.filter(c=>c.support_status==="contradicted").length,
    citation_validity:null};
};
const clinicalEmpty = () => '<div class="dropzone"><strong>No verified RasaChatbot evaluations imported</strong><p>Import de-identified JSON or JSONL evaluation runs matching the documented schema. The browser does not upload or store imported clinical records. Repository code and evaluation harnesses are linked, but do not constitute completed experimental measurements.</p><label for="evaluation-file" class="button">Import evaluation JSON / JSONL</label><input class="inputfile" id="evaluation-file" type="file" accept=".json,.jsonl,application/json"></div>';
const runTable = (runs) => '<div class="table-scroll"><table><thead><tr><th>Scenario / run</th><th>Variant</th><th>Outcome</th><th>Turns</th><th>Latency</th><th>Tokens</th><th>Cost (USD)</th></tr></thead><tbody>' + runs.map(r=>
  '<tr class="selectable" data-run="' + esc(r.run_id) + '" role="button" tabindex="0" aria-label="Inspect evaluation run"><td class="title-cell">' + esc(r.scenario_id) + '<div class="fine">' + esc(r.run_id) + '</div></td><td>' + badge(r.variant) + '</td><td>' + badge(r.outcome || "not_recorded") + '</td><td>' + num(r.turns.length) + '</td><td>' + (r.latency_ms===null?"Unknown":num(r.latency_ms)+" ms") + '</td><td>' + num(r.total_tokens) + '</td><td>' + (r.cost_usd===null?"Unknown":r.cost_usd.toFixed(4)) + '</td></tr>').join("")+'</tbody></table></div>';
const runDetail = (run) => {
  const claims = run.turns.flatMap((t,i)=>(t.claims || []).map(c=>({...c,turn:i+1})));
  return '<button class="button secondary small" id="back-runs">← All scenarios</button><div class="section"><div class="panel">' +
    sec("Turn-by-turn clinical decision timeline",run.scenario_id+" / "+run.variant) +
    '<div class="timeline">' + run.turns.map((t,i)=>'<div class="timeline-event"><strong>Turn '+(i+1)+' · '+esc(t.role || "unknown role")+'</strong><p>'+esc(t.message || "Message not recorded")+'</p><div class="pill-row">'+badge(t.workflow || "routing unknown")+badge(t.decision || "decision unknown")+badge(t.validation_status || "validation unknown")+'</div>' +
    '<div class="detail-grid">' + entry("State before",t.state_before===null?null:JSON.stringify(t.state_before)) + entry("State delta",t.state_delta===null?null:JSON.stringify(t.state_delta)) + entry("Selected tool",t.tool) + entry("Retrieval backend",t.retrieval_backend) + entry("Clarification / referral",t.clarification_or_referral) + entry("Citation provenance",t.provenance?.map(p=>p.source_id || p.url || p.chunk_id).join(", ") || null) + '</div></div>').join("")+'</div></div></div>' +
    '<div class="section"><div class="panel">' + sec("Claim-level support audit","Support classification must be supplied by an external grader or clinical reviewer") +
    (claims.length ? '<div class="table-scroll"><table><thead><tr><th>Turn</th><th>Claim</th><th>Support</th><th>Evidence IDs</th><th>Review</th></tr></thead><tbody>'+claims.map(c=>'<tr><td>'+c.turn+'</td><td class="title-cell">'+esc(c.text || "Unknown claim")+'</td><td>'+badge(c.support_status)+'</td><td>'+esc((c.evidence_ids||[]).join(", ") || "Unknown")+'</td><td>'+badge(c.clinical_review_flag || "unreviewed")+'</td></tr>').join("")+'</tbody></table></div>' : empty("No claim audit attached","No claim-level support evidence was included in the imported run.")) +
    '</div></div>' +
    '<div class="section"><div class="panel">' + sec("Clinical reviewer flags & errors","Do not interpret an empty field as pass") +
    '<div class="detail-grid">' + entry("Clinical review status",run.clinical_review_status) + entry("Validation errors",(run.validation_errors||[]).join("; ") || null) + entry("Contradictions",(run.contradictions||[]).join("; ") || null) + entry("Stress category",run.stress_category) + entry("Ablations",(run.ablations||[]).join(", ") || null) + entry("SQL-first retrieval observed",run.turns.some(t=>t.retrieval_backend==="sql_fts") ? "At least one SQL/FTS turn" : "Not confirmed") + '</div></div></div>';
};
const renderClinical = () => {
  const runs = state.evalRuns.filter(r=>state.scenarioFilter==="all" || r.variant===state.scenarioFilter);
  const m = runMetrics(runs);
  const selected = state.evalRuns.find(r=>r.run_id===state.selectedRun);
  return head("RasaChatbot / clinical evaluation","Multi-turn clinical evaluations","Separate from the production chat UI. Import de-identified, real evaluation traces for decision routing, clarification/referral, state changes, SQL-first evidence retrieval, provenance and claim support.",
    '<div class="linklist">' + link("https://github.com/Armanasq/rasachatbot-backend","Backend") + link("https://github.com/Armanasq/rasachatbot-ui","Existing UI") + '</div>') +
    '<div class="stats">' + stat("Evaluation runs",runs.length,"Verified imported records only") + stat("p50 latency (ms)",m.p50,"Run-level metric; no interpolation") + stat("p95 latency (ms)",m.p95,"Sample quantile; only recorded values") + stat("Unsupported claims",m.claims ? m.unsupported : null,"Only claims with grading records") + '</div>' +
    '<div class="section"><div class="panel">' + sec("Evaluation imports","Schema-validated, in-memory only") + clinicalEmpty() +
    '<div class="pill-row"><button class="button outline small" id="download-schema">Download JSON template</button><button class="button outline small" id="export-eval">Export loaded JSON</button><button class="button outline small" id="clear-eval"'+(state.evalRuns.length?"":" disabled")+'>Clear imported runs</button></div></div></div>' +
    '<div class="section">' + sec("Scenario inventory","Select a run for the turn timeline and claim-by-claim audit") +
    '<div class="panel tight"><div class="toolbar">' + select("scenario-filter",state.scenarioFilter,[["all","All variants"],...new Set(state.evalRuns.map(r=>r.variant))].map(x=>Array.isArray(x)?x:[x,x])) +
    '<span class="hint">'+num(runs.length)+' imported runs</span></div>' +
    (runs.length ? runTable(runs) : empty("No observed runs","The production backend may have test fixtures and logs, but no validated telemetry export has been imported into this page.")) + '</div></div>' +
    (selected ? '<div class="section">'+runDetail(selected)+'</div>' : '') +
    '<div class="section"><div class="note warn-note">Clinical safety gate: importing a run is not clinical validation. Unsupported claims, contradictory evidence, referral/triage errors and unresolved safety flags require qualified human review. Never import protected patient information into a public web application.</div></div>';
};
const renderExperiments = () => {
  const variants = [...new Set(state.evalRuns.map(r=>r.variant))];
  const grouped = variants.map(v=>({variant:v,runs:state.evalRuns.filter(r=>r.variant===v)}));
  const metricRows = grouped.map(x=>({variant:x.variant,count:x.runs.length,...runMetrics(x.runs)}));
  const stress = [...new Set(state.evalRuns.map(r=>r.stress_category).filter(Boolean))];
  const abl = [...new Set(state.evalRuns.flatMap(r=>r.ablations))];
  const groupTable = '<div class="table-scroll"><table><thead><tr><th>Variant</th><th>Runs</th><th>p50 latency (ms)</th><th>p95 latency (ms)</th><th>Tokens</th><th>Cost (USD)</th><th>Unsupported claims</th><th>Contradicted</th></tr></thead><tbody>' +
    metricRows.map(r=>'<tr><td>'+badge(r.variant)+'</td><td>'+num(r.count)+'</td><td>'+num(r.p50)+'</td><td>'+num(r.p95)+'</td><td>'+num(r.tokens)+'</td><td>'+(r.cost===null?"Unknown":r.cost.toFixed(4))+'</td><td>'+num(r.claims?r.unsupported:null)+'</td><td>'+num(r.claims?r.contradicted:null)+'</td></tr>').join("")+'</tbody></table></div>';
  return head("RasaChatbot / experimental comparisons","Baselines, ablations & stress tests","Variant comparisons are populated only from imported observations. A comparison must use matched scenario IDs, comparable protocols, and consistent denominators; no performance advantage is presumed.") +
  '<div class="section">'+sec("Baseline vs BACIE / COTAR","Observed metrics only; matching scenario coverage must be checked before interpreting deltas")+
  '<div class="panel tight">'+(metricRows.length?groupTable:empty("No evaluated variants","Import measured baseline, BACIE and COTAR runs with matching scenario IDs to populate this comparison."))+'</div></div>'+
  '<div class="section"><div class="two-col"><div class="panel">'+sec("Ablation dimensions","Tags from imported runs")+(abl.length?bars(abl.map(a=>[a,state.evalRuns.filter(r=>r.ablations.includes(a)).length])):empty("No ablation observations","Model, state, routing, retrieval, judge, and context-budget interventions may be tracked using explicit ablation labels."))+'</div><div class="panel">'+sec("Stress-test coverage","Long context, multi-patient, contradictions, failure injection")+(stress.length?bars(stress.map(a=>[a,state.evalRuns.filter(r=>r.stress_category===a).length])):empty("No stress-test observations","Stress coverage remains unknown until named scenarios are executed and their results exported."))+'</div></div></div>'+
  '<div class="section"><div class="panel">'+sec("Interpretation and provenance gates","Avoid misleading comparisons")+
  '<div class="detail-grid">'+entry("Matched scenario IDs",metricRows.length>=2?"Requires pair-level verification":"Unknown")+
  entry("Citation validity",state.evalRuns.length?"Requires claim/citation-level grader":"Unknown")+
  entry("Clinical review flags",state.evalRuns.length?state.evalRuns.filter(r=>r.clinical_review_status!=="approved").length+" non-approved or unreviewed":"Unknown")+
  entry("Decision-level validation",state.evalRuns.length?state.evalRuns.filter(r=>r.validation_errors.length>0).length+" runs report errors":"Unknown")+
  '</div><p class="fine">Reported latency quantiles are across runs with observed timestamps. Token and dollar totals exclude missing observations. Such groups are not automatically comparable across protocols, contexts, judges or case distributions.</p></div></div>';
};
const renderRepo = () => {
  const repos = state.snapshot.repositories || {};
  return head("Operations / provenance & synchronization","Repository status & synchronization","Source branches remain authoritative. The public site checks for newly published snapshots each minute; private APEX-G changes become visible only after source compilation and authorized mirror publication.",
    link("https://github.com/Armanasq/APEX-G/actions","GitHub Actions","button outline"))+
    '<div class="sync-ledger"><strong>Live synchronization status</strong><p id="sync-detail">Fetching published source version…</p><p class="fine">Polling is automatic while this tab is visible. This public mirror cannot securely read private repositories directly; source publishing depends on the configured GitHub Actions credential. Staleness is reported rather than silently treated as up to date.</p></div>' +
    '<div class="section">'+sec("Connected repositories","Commit history only where authenticated Actions snapshot access was available")+
    '<div class="three-col">'+Object.entries(repos).map(([name,r])=>'<div class="panel"><div class="repo-card"><div><div class="repo-title">'+esc(name)+'</div><div class="repo-url">main · '+esc(r.status)+'</div></div>'+badge(r.status)+'</div><p class="fine">'+esc(r.reason || "GitHub commit feed in build-time snapshot")+'</p>'+link(r.url,"Open repository")+(r.build_commit?'<p class="fine">Built from '+esc(r.build_commit.slice(0,12))+'</p>':"")+'</div>').join("")+'</div></div>'+
    '<div class="section"><div class="panel">'+sec("Main-branch commit timeline","Read-only repository events; publication may lag the live branch")+
    (Object.entries(repos).some(([name,r])=>r.commits?.length)?'<div class="timeline">'+Object.entries(repos).flatMap(([name,r])=>(r.commits||[]).map(c=>({name,...c}))).sort((a,b)=>(b.date||"").localeCompare(a.date||"")).slice(0,30).map(c=>'<div class="timeline-event"><strong>'+esc(c.name)+'</strong> '+link(c.url,c.sha.slice(0,9),"")+'<p>'+esc(c.message)+' · '+esc(c.date||"date unknown")+'</p></div>').join("")+'</div>':empty("Commit metadata unavailable","The build needs read access to each private repository. Missing history is an authentication or permission limitation, not proof of repository inactivity."))+
    '</div></div>'+
    '<div class="section"><div class="two-col"><div class="panel">'+sec("Synchronization semantics","Unidirectional, source-controlled publication")+
    '<div class="timeline"><div class="timeline-event"><strong>APEX-G main</strong><p>Authoritative CSV manifests, processing inventory, Markdown records and synthesis.</p></div><div class="timeline-event"><strong>Validate & compile</strong><p>Python reconciles aliases and source counts, creates a sanitized JSON snapshot and fails on inconsistent archive totals.</p></div><div class="timeline-event"><strong>GitHub Pages</strong><p>Read-only static app with local evaluation imports and no secret-bearing client requests.</p></div></div></div>'+
    '<div class="panel">'+sec("README & workflow entry points","Keep the research corpus and app separable")+
    '<div class="summary-links">'+link("https://github.com/Armanasq/APEX-G/blob/main/README.md","APEX-G README","summary-link")+
    link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/README.md","Literature README","summary-link")+
    link("https://github.com/Armanasq/APEX-G/blob/main/research-workspace/README.md","Workspace README","summary-link")+
    link("https://github.com/Armanasq/APEX-G/actions/workflows/research-workspace-pages.yml","Pages build workflow","summary-link")+
    link("https://github.com/Armanasq/rasachatbot-backend","Rasa backend","summary-link")+
    link("https://github.com/Armanasq/rasachatbot-ui","Rasa frontend","summary-link")+'</div></div></div></div>'+
    '<div class="section"><div class="panel">'+sec("Source lineage","Build-time auditable provenance")+
    '<div class="table-scroll"><table><thead><tr><th>Source path</th><th>Role</th><th>Open</th></tr></thead><tbody>'+
    state.snapshot.provenance.map(x=>'<tr><td class="title-cell">'+esc(x.path)+'</td><td>'+esc(x.kind.replaceAll("_"," "))+'</td><td>'+link("https://github.com/Armanasq/APEX-G/blob/main/"+x.path,"View")+'</td></tr>').join("")+'</tbody></table></div></div></div>';
};
const template = () => ({
  schema_version:"1.0.0",
  runs:[{
    run_id:"replace-with-unique-run-id",scenario_id:"replace-with-scenario-id",variant:"baseline",
    outcome:null,latency_ms:null,total_tokens:null,cost_usd:null,
    clinical_review_status:"unreviewed",validation_errors:[],contradictions:[],stress_category:null,ablations:[],
    turns:[{role:"patient",message:"De-identified evaluation text",workflow:null,decision:null,
      tool:null,state_before:null,state_delta:null,clarification_or_referral:null,retrieval_backend:null,
      provenance:[],claims:[],validation_status:null}]
  }]
});
const allowedVariants = ["baseline","BACIE","COTAR"];
const normalizeRun = (r,i) => {
  if (!r || typeof r!=="object" || Array.isArray(r)) throw Error("Run "+(i+1)+" is not an object");
  if (typeof r.run_id!=="string" || !r.run_id.trim() || typeof r.scenario_id!=="string" || !r.scenario_id.trim()) throw Error("Run "+(i+1)+" must include run_id and scenario_id");
  if (typeof r.variant!=="string" || !r.variant.trim()) throw Error("Run "+(i+1)+" must include variant");
  if (!Array.isArray(r.turns)) throw Error("Run "+(i+1)+" must include a turns array");
  const optionalNum = (v,k) => {if(v===undefined||v===null)return null;if(typeof v!=="number"||!Number.isFinite(v)||v<0)throw Error("Invalid "+k+" in run "+(i+1));return v;};
  const strArray = (v,key) => {if(v===null||v===undefined)return [];if(!Array.isArray(v)||!v.every(x=>typeof x==="string"))throw Error("Invalid "+key);return v;};
  return {
    run_id:r.run_id,scenario_id:r.scenario_id,variant:r.variant,outcome:safeString(r.outcome),
    latency_ms:optionalNum(r.latency_ms,"latency_ms"),total_tokens:optionalNum(r.total_tokens,"total_tokens"),
    cost_usd:optionalNum(r.cost_usd,"cost_usd"),clinical_review_status:safeString(r.clinical_review_status)||"unreviewed",
    validation_errors:strArray(r.validation_errors,"validation_errors"),contradictions:strArray(r.contradictions,"contradictions"),
    stress_category:safeString(r.stress_category),ablations:strArray(r.ablations,"ablations"),
    turns:r.turns.map((t,j)=>{
      if(!t||typeof t!=="object"||Array.isArray(t))throw Error("Invalid turn "+(j+1));
      const claims = Array.isArray(t.claims)?t.claims:[];
      const provenance = Array.isArray(t.provenance)?t.provenance:[];
      return {role:safeString(t.role),message:safeString(t.message),workflow:safeString(t.workflow),decision:safeString(t.decision),
        tool:safeString(t.tool),state_before:t.state_before??null,state_delta:t.state_delta??null,
        clarification_or_referral:safeString(t.clarification_or_referral),retrieval_backend:safeString(t.retrieval_backend),
        validation_status:safeString(t.validation_status),
        provenance:provenance.filter(p=>p&&typeof p==="object").map(p=>({source_id:safeString(p.source_id),url:safeString(p.url),chunk_id:safeString(p.chunk_id)})),
        claims:claims.filter(c=>c&&typeof c==="object").map(c=>({text:safeString(c.text),support_status:safeString(c.support_status)||"unreviewed",evidence_ids:strArray(c.evidence_ids,"evidence_ids"),clinical_review_flag:safeString(c.clinical_review_flag)}))};
    }),
  };
};
const importRuns = async file => {
  if (!file) return;
  if (file.size > 8*1024*1024) throw Error("Evaluation file too large (8 MiB limit)");
  const text = await file.text();
  const incoming = file.name.toLowerCase().endsWith(".jsonl") ? text.split(/\r?\n/).filter(Boolean).map((line,i)=>{try{return JSON.parse(line)}catch{throw Error("Invalid JSONL at line "+(i+1))}}) : JSON.parse(text);
  const raw = Array.isArray(incoming)?incoming:Array.isArray(incoming.runs)?incoming.runs:[incoming];
  if (raw.length>3000) throw Error("Import maximum is 3000 runs");
  const normalized = raw.map(normalizeRun);
  const ids = new Set();
  for (const run of normalized) {if(ids.has(run.run_id))throw Error("Duplicate run_id: "+run.run_id);ids.add(run.run_id);}
  state.evalRuns = normalized;
  state.selectedRun = null;
  state.scenarioFilter = "all";
  state.runImportLabel = file.name;
  notify("Imported "+normalized.length+" evaluation runs. Nothing uploaded.");
  render();
};
const render = () => {
  if (!state.snapshot) return;
  const labels = {overview:"Overview",papers:"Paper registry",evidence:"Evidence explorer",synthesis:"Synthesis & gaps",clinical:"Scenario evaluation",experiments:"Comparisons & stress",repository:"Repositories & sync"};
  $("#breadcrumb").textContent = labels[state.page] || "Overview";
  document.querySelectorAll(".nav-item").forEach(node=>node.classList.toggle("active",node.dataset.page===state.page));
  const views = {overview:renderOverview,papers:state.paper?renderPaperDetail:renderPapers,evidence:renderEvidence,synthesis:renderSynthesis,clinical:renderClinical,experiments:renderExperiments,repository:renderRepo};
  $("#content").innerHTML = (views[state.page] || renderOverview)();
  bind();
};
const go = (page) => {state.page=page;state.paper=null;state.selectedRun=null;history.replaceState(null,"","#"+page);render();$("#content").focus();window.scrollTo({top:0,behavior:"instant"});};
const bind = () => {
  document.querySelectorAll("[data-paper]").forEach(tr=>{const goPaper=()=>{state.paper=tr.dataset.paper;state.page="papers";render();window.scrollTo({top:0,behavior:"instant"});};tr.addEventListener("click",goPaper);tr.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();goPaper();}});});
  document.querySelectorAll("[data-run]").forEach(tr=>{const fn=()=>{state.selectedRun=tr.dataset.run;render();};tr.addEventListener("click",fn);tr.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();fn();}});});
  document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>{go(b.dataset.go);if(b.dataset.status){state.status=b.dataset.status;render();}}));
  const input=$("#paper-search");if(input)input.addEventListener("input",()=>{const cursor=input.selectionStart;state.query=input.value;render();const next=$("#paper-search");if(next){next.focus();next.setSelectionRange(cursor,cursor);}});
  [["topic-filter","topic"],["year-filter","year"],["status-filter","status"],["evidence-field","evidenceField"],["scenario-filter","scenarioFilter"]].forEach(([id,key])=>{const n=$("#"+id);if(n)n.addEventListener("change",()=>{state[key]=n.value;render();});});
  const exportP=()=>htmlDownload("apex-g-filtered-papers.csv",csv(filteredPapers(),["id","title","year","topic","evidence_type","publication_doi","preprint_doi","reading_status","metadata_status","acquisition_status","review_url","publisher_url","pdf_url"]),"text/csv;charset=utf-8");
  ["export-papers","export-papers-top"].forEach(id=>{const n=$("#"+id);if(n)n.addEventListener("click",exportP);});
  const back=$("#back-papers");if(back)back.addEventListener("click",()=>{state.paper=null;render();});
  const file=$("#evaluation-file");if(file)file.addEventListener("change",async e=>{try{await importRuns(e.target.files?.[0]);}catch(err){notify(err.message||"Import failed");}});
  const schema=$("#download-schema");if(schema)schema.addEventListener("click",()=>htmlDownload("rasa-evaluation-template.json",JSON.stringify(template(),null,2),"application/json"));
  const exportE=$("#export-eval");if(exportE)exportE.addEventListener("click",()=>htmlDownload("rasa-evaluations.json",JSON.stringify({schema_version:"1.0.0",runs:state.evalRuns},null,2),"application/json"));
  const clear=$("#clear-eval");if(clear)clear.addEventListener("click",()=>{state.evalRuns=[];state.selectedRun=null;render();notify("Evaluation data cleared from this browser session");});
  const backR=$("#back-runs");if(backR)backR.addEventListener("click",()=>{state.selectedRun=null;render();});
};
const REFRESH_INTERVAL_MS = 60_000;
const MAX_ACCEPTABLE_AGE_MS = 6 * 60 * 60 * 1000;
let lastVersion = null;
let updateInFlight = false;
let lastSuccessfulCheck = null;
let lastRefreshError = null;
const isoTime = value => {
  const parsed = new Date(value || "");
  return Number.isNaN(parsed.getTime()) ? "Unknown" : parsed.toLocaleString();
};
const snapshotIdentity = data => [
  data?.generated_at || "",
  data?.repositories?.["Armanasq/APEX-G"]?.build_commit || "",
].join("|");
const freshness = () => {
  if (!state.snapshot) return;
  const stamp = new Date(state.snapshot.generated_at).getTime();
  const age = Number.isFinite(stamp) ? Date.now() - stamp : NaN;
  const stale = !Number.isFinite(age) || age > MAX_ACCEPTABLE_AGE_MS;
  const label = lastRefreshError ? "Source check failed" : stale ? "Snapshot may be stale" : "Snapshot current";
  const status = document.querySelector("#sync-status");
  if (status) {
    status.textContent = label;
    status.className = "sync-status " + (lastRefreshError ? "sync-error" : stale ? "sync-stale" : "sync-current");
    status.title = "Snapshot built " + isoTime(state.snapshot.generated_at) + (lastSuccessfulCheck ? " · checked " + isoTime(lastSuccessfulCheck) : "");
  }
  const stampNode = document.querySelector("#snapshot-time");
  if (stampNode) {
    stampNode.textContent = "Source snapshot · " + isoTime(state.snapshot.generated_at);
    stampNode.title = "Generated by APEX-G GitHub Actions. This does not guarantee that the public mirror is current with private main.";
  }
  const syncNote = document.querySelector("#sync-detail");
  if (syncNote) syncNote.textContent = "Snapshot: " + isoTime(state.snapshot.generated_at) + " · Latest successful check: " + (lastSuccessfulCheck ? isoTime(lastSuccessfulCheck) : "Not yet checked") + (lastRefreshError ? " · " + lastRefreshError : "");
};
const loadVersion = async () => {
  const response = await fetch("./data/version.json?nonce=" + Date.now(), {cache:"no-store"});
  if (!response.ok) throw Error("Version feed HTTP " + response.status);
  const version = await response.json();
  if (version.schema_version !== "1.0.0" || !version.generated_at) throw Error("Invalid version metadata");
  return version;
};
const checkForUpdates = async () => {
  if (updateInFlight || document.visibilityState === "hidden") return;
  updateInFlight = true;
  try {
    const version = await loadVersion();
    lastSuccessfulCheck = new Date().toISOString();
    lastRefreshError = null;
    if (!state.snapshot || lastVersion !== version.generated_at || version.generated_at !== state.snapshot.generated_at) {
      lastVersion = version.generated_at;
      await loadSnapshot();
    }
  } catch (error) {
    lastRefreshError = error.message || "Refresh failed";
  } finally {
    updateInFlight = false;
    freshness();
  }
};
const loadSnapshot = async () => {
  try {
    const response = await fetch("./data/snapshot.json?cache="+Date.now(),{cache:"no-store"});
    if(!response.ok)throw Error("Snapshot fetch failed: HTTP "+response.status);
    const data = await response.json();
    if(data.schema_version!=="1.0.0" || !Array.isArray(data.papers))throw Error("Incompatible snapshot schema");
    const changed = snapshotIdentity(state.snapshot) !== snapshotIdentity(data);
    state.snapshot = data;
    lastVersion = data.generated_at;
    lastSuccessfulCheck = new Date().toISOString();
    lastRefreshError = null;
    if (changed) render();
    freshness();
  }catch(error) {
    if(!state.snapshot)$("#content").innerHTML='<div class="panel">'+empty("Source snapshot not yet generated","The source-controlled build must run scripts/build_snapshot.py to create data/snapshot.json before this app can display repository records.")+'</div>';
    lastRefreshError = error.message || "Snapshot unavailable";
    freshness();
    notify(lastRefreshError);
  }
};
document.querySelectorAll(".nav-item").forEach(b=>b.addEventListener("click",()=>go(b.dataset.page)));
$("#refresh").addEventListener("click", async () => {
  await loadSnapshot();
  checkForUpdates();
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") checkForUpdates();
});
window.setInterval(checkForUpdates, REFRESH_INTERVAL_MS);
window.setInterval(freshness, 30_000);
const initialHash=location.hash.slice(1);
if(["overview","papers","evidence","synthesis","clinical","experiments","repository"].includes(initialHash))state.page=initialHash;
loadSnapshot().then(checkForUpdates);

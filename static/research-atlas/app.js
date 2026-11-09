"use strict";
const allowedPages = new Set(["overview","papers","evidence","synthesis","repository"]);
const defaultPage = "overview";
const DATA_ROOT = "./";
const state = { snapshot: null, page: defaultPage, paper: null, query: "", topic: "all", year: "all", status: "all", evidenceField: "all" };
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
  const summary = state.snapshot.summary || {};
  const ledger = countBy(papers(), "reading_status");
  const indexed = papers().length;
  const completed = ledger.read_complete || 0;
  const visual = ledger.text_complete_visual_pending || 0;
  const notRecorded = ledger.not_recorded || 0;
  const topicCounts = Object.entries(countBy(papers(),"topic_label")).sort((a,b)=>b[1]-a[1]);
  const yearCounts = Object.entries(countBy(papers(),"year")).sort((a,b)=>Number(a[0])-Number(b[0]));
  const queue = papers().filter(p=>p.reading_status !== "read_complete").slice(0,7);
  const reviews = papers().filter(p=>Boolean(p.review_url)).length;
  const percent = value => indexed ? (value / indexed * 100).toFixed(3) : "0";
  const overviewActions = '<button class="button outline" data-go="papers">Open paper registry <span aria-hidden="true">↗</span></button>';
  return head("APEX-G / research record","Research evidence at a glance",
    "Bibliographic coverage, manuscript verification, and the work still required to establish defensible scientific conclusions.", overviewActions) +
    '<div class="editorial-summary">' +
    '<div class="summary-main"><span class="editorial-label">CURRENT CORPUS · SOURCE CONTROLLED</span>' +
    '<p class="editorial-intro">A working account of what is known, what is reviewed, and what remains unresolved.</p>' +
    '<span class="summary-provenance">Source compiled ' + esc(isoTime(state.snapshot.generated_at)) + '</span></div>' +
    '<div class="editorial-meta"><span class="meta-kicker">INDEXED STUDIES</span><strong>' + num(indexed) + '</strong><span>canonical records</span></div></div>' +
    '<section class="review-ledger" aria-label="Scientific verification coverage">' +
    '<div class="review-ledger-top"><div><span class="mini-label">VERIFICATION DISTRIBUTION</span><h2>Reading progress</h2></div>' +
    '<span class="review-denominator">' + num(indexed) + ' indexed records</span></div>' +
    '<div class="review-progress" role="img" aria-label="'+esc(num(completed)+' read complete, '+num(visual)+' awaiting figure verification, '+num(notRecorded)+' not recorded')+'">' +
    '<span class="segment-complete" style="width:'+percent(completed)+'%"></span>' +
    '<span class="segment-visual" style="width:'+percent(visual)+'%"></span>' +
    '<span class="segment-unrecorded" style="width:'+percent(Math.max(0,indexed-completed-visual))+'%"></span></div>' +
    '<div class="review-legend"><div><i class="legend-dot complete"></i><span>Read complete</span><strong>'+num(completed)+'</strong></div>' +
    '<div><i class="legend-dot visual"></i><span>Visual checks pending</span><strong>'+num(visual)+'</strong></div>' +
    '<div><i class="legend-dot unrecorded"></i><span>Other / not recorded</span><strong>'+num(Math.max(0,indexed-completed-visual))+'</strong></div></div></section>' +
    '<div class="stats">' +
      stat("Archived source PDFs",summary.pdfs_reported,"Reported in integrity audit") +
      stat("Linked reviews",reviews,"Recorded Markdown analyses") +
      stat("Acquisition exceptions",state.snapshot.acquisition_queue?.length ?? null,"Recorded manual queue") +
      stat("Unresolved review states",Math.max(0,indexed-completed),"Not equivalent to missing PDFs") +
    '</div>' +
    '<div class="section"><div class="two-col"><div class="panel">' +
    sec("Research domains","Distribution of indexed studies") + bars(topicCounts.slice(0,9)) +
    '</div><div class="panel">' + sec("Publication timeline","Indexed publication years") +
    bars(yearCounts,"teal") + '</div></div></div>' +
    '<div class="section">' + sec("Next to verify","Records that have not passed the complete scientific reading gate",
      '<button class="button secondary small" data-go="papers" data-status="missing">Inspect verification queue →</button>') +
    '<div class="panel tight">' + (queue.length ? paperTable(queue) :
      empty("No pending records","All indexed records currently pass the recorded reading gate.")) + '</div></div>' +
    '<div class="section research-footnote"><span>PROVENANCE NOTE</span><p>Recorded source integrity is not independent manuscript verification. Reading and figure statuses follow the author-maintained APEX-G inventory.</p></div>';
};
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
    '<div class="panel"><div class="eyebrow">Controlling research question</div><h2 style="font-size:19px;line-height:1.5;font-weight:670;max-width:950px">' + fmt(state.snapshot.research_question) + '</h2><div class="note warn-note">The central claim of broadly transferable, persistent latent intelligence growth remains an open empirical question. Benchmark improvement alone is not causal evidence of general intelligence.</div></div>' +
    '<div class="section">' + sec("Evidence-to-synthesis graph","Source files form the audit trail; selection opens upstream records") +
    '<div class="three-col"><div class="panel"><div class="eyebrow">01 · INPUT</div><h3>Primary literature</h3><p class="fine">' + num(papers().length) + ' canonical records · ' + num(papers().filter(p=>p.review_url).length) + ' linked reviews</p><button class="button secondary small" data-go="papers">Inspect paper registry</button></div>' +
    '<div class="panel"><div class="eyebrow">02 · EVIDENCE</div><h3>Methods, claims & counterevidence</h3><p class="fine">Source-linked excerpts, outstanding figure review, competing explanations and limitations.</p><button class="button secondary small" data-go="evidence">Inspect evidence</button></div>' +
    '<div class="panel"><div class="eyebrow">03 · OUTPUT</div><h3>Working scholarly synthesis</h3><p class="fine">Argument synthesis and research gaps, without auto-promoting tentative findings.</p>' + link(assets.working_review,"Open review") + '</div></div></div>' +
    '<div class="section"><div class="two-col"><div class="panel">' + sec("Topic map","Canonical paper records") + bars(Object.entries(counts).sort((a,b)=>b[1]-a[1])) + '</div><div class="panel">' + sec("Canonical working documents","Linked Markdown research assets") +
    '<div class="summary-links">' + Object.entries(labels).map(([key,name])=>link(assets[key],name,"summary-link")).join("") + '</div></div></div></div>' +
    '<div class="section"><div class="panel">' + sec("Missing-paper detection strategy","Two distinct failure modes are preserved") +
    '<div class="two-col"><div><h3>Acquisition gap</h3><p class="fine">A registered paper has no validated downloadable source, requires manual publisher access or has licensing restrictions. Do not treat a link-only source as a failed scientific review.</p><p>' + link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/sources/manual_acquisition_required.csv","Manual queue") + link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/sources/restricted_or_link_only.csv","License restrictions") + '</p></div><div><h3>Coverage gap</h3><p class="fine">A research construct, comparison, publication or negative-control family is absent from the review scope. Candidate detection depends on external bibliographic discovery and independent reconciliation.</p><p>' + link(assets.missing_literature,"Missing-literature audit") + link(assets.gaps,"Research gaps") + '</p></div></div></div></div>';
};
const renderRepo = () => {
  const name = "Armanasq/APEX-G";
  const record = state.snapshot.repositories?.[name] || {};
  const commits = Array.isArray(record.commits) ? record.commits.slice(0, 30) : [];
  return head("APEX-G / repository","Repository & synchronization",
    "APEX-G main is the only source of scientific records. Published snapshots may lag behind source commits.",
    link("https://github.com/Armanasq/APEX-G/actions","Build history","button outline")) +
    '<div class="sync-ledger"><strong>Published data status</strong><p id="sync-detail">Reading current snapshot…</p><p class="fine">The browser checks the public version feed every minute. It does not directly access the private source repository.</p></div>' +
    '<div class="section">' + sec("Source repository","Only APEX-G commit data is displayed") +
    '<div class="panel"><div class="repo-card"><div><div class="repo-title">'+esc(name)+'</div><div class="repo-url">'+esc(record.branch || "main")+' · '+esc(record.status || "unavailable")+'</div></div>'+badge(record.status)+'</div>'+
    '<p class="fine">'+esc(record.reason || "Commit feed compiled from APEX-G main")+'</p>'+link(record.url || "https://github.com/Armanasq/APEX-G","View repository")+
    (record.build_commit ? '<p class="fine">Built from '+esc(record.build_commit.slice(0,12))+'</p>' : "")+
    '</div></div>' +
    '<div class="section"><div class="panel">' + sec("Commit history","Read-only commits from source main") +
    (commits.length ? '<div class="timeline">'+commits.map(item=>'<div class="timeline-event"><strong>'+esc(name)+'</strong> '+link(item.url,item.sha.slice(0,9),"")+'<p>'+esc(item.message)+' · '+esc(item.date||"date unknown")+'</p></div>').join("")+'</div>' :
      empty("Commit feed unavailable","No authenticated commit history was included in the source snapshot.")) +
    '</div></div>' +
    '<div class="section"><div class="two-col"><div class="panel">' + sec("Data publication","From main to the research workspace") +
    '<div class="timeline"><div class="timeline-event"><strong>APEX-G main</strong><p>Bibliography, archive manifests, individual paper reviews and synthesis.</p></div><div class="timeline-event"><strong>Validate & compile</strong><p>Conservative, source-attributed JSON snapshots are generated.</p></div><div class="timeline-event"><strong>Public mirror</strong><p>The authorized deployment publishes APEX-G records only.</p></div></div></div>' +
    '<div class="panel">' + sec("Canonical research documents","Primary research source-of-truth records") +
    '<div class="summary-links">' +
    link("https://github.com/Armanasq/APEX-G/blob/main/README.md","Repository README","summary-link") +
    link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/papers/INDEX.md","Critical paper reviews","summary-link") +
    link("https://github.com/Armanasq/APEX-G/blob/main/literature_review/manuscript/literature_review.md","Integrated manuscript review","summary-link") +
    link("https://github.com/Armanasq/APEX-G/blob/main/research-workspace/README.md","Workspace documentation","summary-link") +
    link("https://github.com/Armanasq/APEX-G/actions/workflows/research-workspace-pages.yml","Build workflow","summary-link") +
    '</div></div></div></div>' +
    '<div class="section"><div class="panel">' + sec("Source lineage","Auditable source files from the literature inventory") +
    '<div class="table-scroll"><table><thead><tr><th>Source path</th><th>Role</th><th>Open</th></tr></thead><tbody>' +
    (state.snapshot.provenance || []).map(item=>'<tr><td class="title-cell">'+esc(item.path)+'</td><td>'+esc(item.kind.replaceAll("_"," "))+'</td><td>'+link("https://github.com/Armanasq/APEX-G/blob/main/"+item.path,"View")+'</td></tr>').join("") +
    '</tbody></table></div></div></div>';
};
const render = () => {
  if (!state.snapshot) return;
  const labels = {overview:"Overview",papers:"Paper registry",evidence:"Evidence explorer",synthesis:"Synthesis & gaps",repository:"Repository & sync"};
  if (!allowedPages.has(state.page)) state.page = defaultPage;
  $("#breadcrumb").textContent = labels[state.page] || labels[defaultPage];
  document.querySelectorAll(".nav-item").forEach(node=>{const active=node.dataset.page===state.page;node.classList.toggle("active",active);if(active)node.setAttribute("aria-current","page");else node.removeAttribute("aria-current");});
  const views = {overview:renderOverview,papers:state.paper?renderPaperDetail:renderPapers,evidence:renderEvidence,synthesis:renderSynthesis,repository:renderRepo};
  $("#content").innerHTML = (views[state.page] || views[defaultPage])();
  bind();
};
const go = (page) => {if(!allowedPages.has(page))return;state.page=page;state.paper=null;history.replaceState(null,"","#"+page);render();$("#content").focus();window.scrollTo({top:0,behavior:"instant"});};
const bind = () => {
  document.querySelectorAll("[data-paper]").forEach(tr=>{const goPaper=()=>{state.paper=tr.dataset.paper;state.page="papers";render();window.scrollTo({top:0,behavior:"instant"});};tr.addEventListener("click",goPaper);tr.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();goPaper();}});});
  document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>{go(b.dataset.go);if(b.dataset.status){state.status=b.dataset.status;render();}}));
  const input=$("#paper-search");if(input)input.addEventListener("input",()=>{const cursor=input.selectionStart;state.query=input.value;render();const next=$("#paper-search");if(next){next.focus();next.setSelectionRange(cursor,cursor);}});
  [["topic-filter","topic"],["year-filter","year"],["status-filter","status"],["evidence-field","evidenceField"]].forEach(([id,key])=>{const n=$("#"+id);if(n)n.addEventListener("change",()=>{state[key]=n.value;render();});});
  const exportP=()=>htmlDownload("apex-g-filtered-papers.csv",csv(filteredPapers(),["id","title","year","topic","evidence_type","publication_doi","preprint_doi","reading_status","metadata_status","acquisition_status","review_url","publisher_url","pdf_url"]),"text/csv;charset=utf-8");
  ["export-papers","export-papers-top"].forEach(id=>{const n=$("#"+id);if(n)n.addEventListener("click",exportP);});
  const back=$("#back-papers");if(back)back.addEventListener("click",()=>{state.paper=null;render();});
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
  const label = lastRefreshError ? "Refresh unavailable" : stale ? "Published data is older" : "Published snapshot loaded";
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
  const response = await fetch(DATA_ROOT+"version.json?nonce=" + Date.now(), {cache:"no-store"});
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
    const response = await fetch(DATA_ROOT+"snapshot.json?cache="+Date.now(),{cache:"no-store"});
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
    if (!state.snapshot) $("#content").innerHTML = '<div class="panel">' + empty("Published project data unavailable", "The project-local data endpoint could not be loaded. Check the deployed snapshot.json asset and GitHub Pages workflow.") + '</div>';
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
if(allowedPages.has(initialHash))state.page=initialHash;
const boot = window.__RESEARCH_ATLAS_BOOTSTRAP__;
if (boot && boot.schema_version === "1.0.0" && Array.isArray(boot.papers)) {
  state.snapshot = boot;
  lastVersion = boot.generated_at;
  render();
  freshness();
  checkForUpdates();
} else {
  loadSnapshot().then(checkForUpdates);
}

// Knowledge Base — browser uploads, real ingestion stages, library and retrieval probe.

const KB_STAGE_ORDER = ["upload", "discover", "parse", "chunk", "embed", "index", "ready"];
const KB_EVENT_STAGE = {
  uploaded:"discover", discovered:"parse", loading:"parse", chunking:"chunk",
  embedding:"embed", indexing:"index", created:"ready", updated:"ready", skipped:"ready",
};
let kbState = {
  busy:false, stage:"", failed:false, filename:"", message:"",
  recent:[], selected:null, chunks:[], chunksLoading:false, catFilter:"all", catEdit:null,
  query:"", hits:[], searching:false,
  evalSet:"", evalRunning:false, evalResult:null, evalError:"", evalUseLlm:false, evalRerank:null, evalWorkspace:"", evalK:6, evalNote:"",
};

function kbRerender(){
  if (activeView === "documents" && D){
    document.getElementById("view").innerHTML = documentsView(D);
  }
}

function kbBytes(value){
  const n = Number(value || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024*1024) return `${(n/1024).toFixed(1)} KB`;
  return `${(n/1024/1024).toFixed(1)} MB`;
}

function kbStageClass(stage){
  if (!kbState.stage) return "";
  const here = KB_STAGE_ORDER.indexOf(stage), current = KB_STAGE_ORDER.indexOf(kbState.stage);
  if (kbState.failed && here === current) return "failed";
  if (here < current || (stage === "ready" && here === current)) return "done";
  return here === current ? "active" : "";
}

function kbFlow(){
  const labels = {
    upload:["Upload","receive bytes"], discover:["Discover","validate source"],
    parse:["Parse","extract safe text"], chunk:["Chunk","preserve context"],
    embed:["Embed","create vectors"], index:["Index","FTS5 + vectors"],
    ready:["Ready","available to agent"],
  };
  return `<div class="kb-flow" aria-label="Ingestion stages">${KB_STAGE_ORDER.map((key, i) => {
    const [name, detail] = labels[key];
    return `${i?'<span class="kb-flow-line"></span>':""}<div class="kb-stage ${kbStageClass(key)}">
      <span class="kb-stage-dot"></span><b>${name}</b><small>${detail}</small></div>`;
  }).join("")}</div>`;
}

function kbPick(kind){
  const input = document.getElementById(kind === "folder" ? "kb-folder-input" : "kb-file-input");
  if (input) input.click();
}

function kbPicked(input){
  const files = Array.from(input.files || []);
  input.value = "";
  kbIngestQueue(files);
}

function kbDrag(event, over){
  event.preventDefault();
  const drop = document.getElementById("kb-drop");
  if (drop) drop.classList.toggle("over", !!over);
}

function kbDrop(event){
  kbDrag(event, false);
  kbIngestQueue(Array.from(event.dataTransfer.files || []));
}

async function kbIngestQueue(files){
  if (kbState.busy || !files.length) return;
  kbState.busy = true; kbState.failed = false; kbState.recent = [];
  for (const file of files){
    kbState.failed = false;
    kbState.filename = file.webkitRelativePath || file.name;
    kbState.stage = "upload"; kbState.message = `Uploading ${file.name}`; kbRerender();
    try {
      const workspace = (D.documents && D.documents.workspace) || "default";
      const uploadName = file.webkitRelativePath || file.name;
      const url = `/api/documents/ingest?name=${encodeURIComponent(uploadName)}&workspace=${encodeURIComponent(workspace)}`;
      const response = await fetch(url, {method:"POST", headers:{"Content-Type":"application/octet-stream"}, body:file});
      if (!response.ok){
        let detail = `Upload failed (${response.status})`;
        try { detail = (await response.json()).error || detail; } catch(e){}
        throw new Error(detail);
      }
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = "", finalEvent = null;
      for (;;){
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream:true});
        let split;
        while ((split = buffer.indexOf("\n\n")) >= 0){
          const line = buffer.slice(0, split); buffer = buffer.slice(split + 2);
          if (!line.startsWith("data:")) continue;
          let event; try { event = JSON.parse(line.slice(5).trim()); } catch(e){ continue; }
          if (event.kind === "done") finalEvent = event;
          else {
            kbState.stage = KB_EVENT_STAGE[event.kind] || kbState.stage;
            kbState.message = `${file.name} · ${event.kind}`;
            kbRerender();
          }
        }
      }
      if (!finalEvent || finalEvent.error) throw new Error(finalEvent?.error || "Ingestion ended without a result");
      const report = (finalEvent.reports || [])[0];
      if (!report) throw new Error("Ingestion returned no report");
      if (report.action === "failed") throw new Error(report.error || "Ingestion failed");
      kbState.stage = "ready";
      kbState.message = `${report.title} · ${report.action} · ${report.chunk_count} chunks`;
      kbState.recent.unshift({ok:true, name:file.name, detail:kbState.message});
    } catch(error){
      kbState.failed = true;
      kbState.message = `${file.name} · ${error.message || error}`;
      kbState.recent.unshift({ok:false, name:file.name, detail:String(error.message || error)});
    }
    kbRerender();
  }
  kbState.busy = false;
  editing = false;
  await refresh();
}

async function kbSearch(){
  const input = document.getElementById("kb-query");
  const query = (input && input.value || "").trim();
  if (!query) return;
  kbState.query = query; kbState.searching = true; kbState.hits = []; kbRerender();
  const workspace = (D.documents && D.documents.workspace) || "default";
  const result = await postJSON("/api/documents/search", {query, workspace, top_k:6});
  kbState.searching = false;
  kbState.hits = result.hits || [];
  kbState.message = result.error || (kbState.hits.length ? "" : "No indexed chunk matched that query.");
  editing = false; kbRerender();
}

function kbSearchKey(event){ if (event.key === "Enter") kbSearch(); }

function kbEvalPick(select){
  // Re-render so the Run button enables immediately, and release the
  // auto-refresh edit-lock the select took on focus (see markEditing).
  kbState.evalSet = select.value; kbState.evalResult = null; kbState.evalError = "";
  editing = false; kbRerender();
}

function kbEvalK(select){ kbState.evalK = Number(select.value) || 6; kbState.evalResult = null; }

function kbEvalWorkspace(input){ kbState.evalWorkspace = input.value.trim(); kbState.evalResult = null; }

function kbEvalRerank(box){ kbState.evalRerank = box.checked; kbState.evalResult = null; editing = false; kbRerender(); }

function kbEvalLlm(box){ kbState.evalUseLlm = box.checked; kbState.evalResult = null; editing = false; kbRerender(); }

function kbEvalImportPick(){ const input = document.getElementById("kb-eval-file"); if (input) input.click(); }

async function kbEvalImport(input){
  const file = (input.files || [])[0];
  input.value = "";
  if (!file) return;
  kbState.evalError = ""; kbState.evalNote = "";
  const result = await postJSON("/api/documents/eval/import", {name:file.name, content:await file.text()});
  if (result.error) kbState.evalError = result.error;
  else { kbState.evalSet = result.name; kbState.evalResult = null; kbState.evalNote = `Imported ${result.name} · ${result.count} questions`; }
  editing = false;
  await refresh();
  kbRerender();
}

async function kbRunEval(){
  if (!kbState.evalSet || kbState.evalRunning) return;
  kbState.evalRunning = true; kbState.evalError = ""; kbState.evalResult = null; kbRerender();
  const workspace = kbState.evalWorkspace || (D.documents && D.documents.workspace) || "default";
  const result = await postJSON("/api/documents/eval", {question_set:kbState.evalSet, workspace, k:kbState.evalK, use_llm:kbState.evalUseLlm, rerank:kbState.evalRerank});
  kbState.evalRunning = false;
  if (result.error) kbState.evalError = result.error;
  else kbState.evalResult = result;
  kbRerender();
}

async function kbViewChunks(documentId){
  kbState.selected = documentId; kbState.chunks = []; kbState.chunksLoading = true; kbRerender();
  const workspace = (D.documents && D.documents.workspace) || "default";
  const response = await fetch(`/api/documents/chunks?document_id=${encodeURIComponent(documentId)}&workspace=${encodeURIComponent(workspace)}`);
  const result = await response.json();
  kbState.chunksLoading = false;
  kbState.chunks = result.chunks || [];
  kbState.message = result.error || "";
  kbRerender();
}

async function kbDocumentAction(action, documentId){
  const document = ((D.documents && D.documents.items) || []).find(item => item.document_id === documentId);
  const title = document ? document.title : "document";
  if (action === "delete" && !confirm(`Remove “${title}” from the index? The original file stays where it is stored (${(document && (document.source_uri||"").startsWith("s3://")) ? "MinIO" : "disk"}).`)) return;
  const workspace = (D.documents && D.documents.workspace) || "default";
  kbState.message = action === "delete" ? `Removing ${title}` : `Re-indexing ${title}`;
  kbState.busy = true; kbState.stage = action === "reindex" ? "index" : ""; kbRerender();
  const result = await postJSON("/api/documents/action", {action, document_id:documentId, workspace});
  kbState.busy = false;
  if (result.error) kbState.message = result.error;
  else {
    kbState.message = action === "delete" ? `${title} removed from the index` : `${title} re-indexed`;
    if (action === "delete" && kbState.selected === documentId){ kbState.selected = null; kbState.chunks = []; }
  }
  await refresh();
}

// --- Document catalog: type / period / status per document, editable ---------
// Editing here is a metadata write (POST /api/documents/catalog): no chunk is
// re-embedded and nothing is OCR'd again, and an edited field is locked against
// later automatic re-classification. Re-index is cheap too: OCR is cached per
// page and vectors per chunk, so only genuinely new content costs anything.
function kbCat(document){ return ((D.documents && D.documents.catalog) || {})[document.source_uri]; }
function kbCatOptions(){ return (D.documents && D.documents.catalog_options) || {types:{}, statuses:{}}; }

function kbCatalogVisible(documents){
  const f = kbState.catFilter;
  if (f === "all") return documents;
  return documents.filter(d => { const c = kbCat(d); return f === "review" ? !!(c && c.needs_review) : !!(c && c.doc_type === f); });
}

function kbCatalogBar(documents){
  const types = kbCatOptions().types, counts = {}, review = documents.filter(d => (kbCat(d) || {}).needs_review).length;
  documents.forEach(d => { const c = kbCat(d); if (c) counts[c.doc_type] = (counts[c.doc_type] || 0) + 1; });
  const chip = (key, label, n) => `<button class="quiet" style="${kbState.catFilter===key?"font-weight:700;text-decoration:underline":""}" onclick="kbCatalogFilter('${key}')">${esc(label)}${n==null?"":` (${n})`}</button>`;
  const uncatalogued = documents.filter(d => !kbCat(d)).length;
  return `<div style="display:flex;flex-wrap:wrap;gap:4px 10px;align-items:center;margin:0 0 8px">
    ${chip("all","All",documents.length)}${Object.keys(types).filter(k => counts[k]).map(k => chip(k, types[k], counts[k])).join("")}
    ${review ? chip("review","Needs review",review) : ""}
    <button class="quiet" style="margin-left:auto" title="Re-ingests every document so each gets a catalog entry. Free for unchanged files: vectors and OCR pages come from the cache." onclick="kbCatalogRefresh()">${uncatalogued ? `Catalogue ${uncatalogued} uncatalogued` : "Refresh catalog"}</button></div>`;
}

function kbCatalogLine(document){
  const c = kbCat(document);
  if (!c) return `<small>not catalogued yet</small>`;
  const types = kbCatOptions().types, statuses = kbCatOptions().statuses;
  const bits = [types[c.doc_type] || c.doc_type, c.subtype, c.period, c.issuer].filter(Boolean).map(esc).join(" · ");
  const status = c.status !== "current" ? ` · <b>${esc(statuses[c.status] || c.status)}</b>` : "";
  const review = c.needs_review ? ` · <b style="color:var(--bad)" title="${esc(c.evidence)} (confidence ${Math.round(c.confidence*100)}%)">needs review</b>` : "";
  return `<small>${bits}${status}${review}</small>`;
}

function kbCatalogEditor(document){
  if (kbState.catEdit !== document.document_id) return "";
  const c = kbCat(document) || {}, id = esc(document.document_id), o = kbCatOptions(), p = c.profile || {};
  const opts = (map, cur) => Object.keys(map).map(k => `<option value="${esc(k)}" ${k===cur?"selected":""}>${esc(map[k])}</option>`).join("");
  const field = (name, label, value) => `<label style="display:grid;gap:2px;font-size:11px">${label}<input id="kb-cat-${name}-${id}" value="${esc(value||"")}" onfocus="markEditing()"></label>`;
  const facts = [p.pages ? `${p.pages} pages` : "", p.language ? `language ${p.language}` : "", p.scanned ? "scanned (OCR)" : "text PDF", p.table_heavy ? "table-heavy" : ""].filter(Boolean).join(" · ");
  return `<div style="padding:8px 10px 10px;border-top:1px solid var(--line);display:grid;gap:8px;background:var(--panel)">
    <small style="color:var(--ink3)">${esc(facts)}${c.evidence ? ` · guessed from ${esc(c.evidence)} (${Math.round((c.confidence||0)*100)}%)` : ""}${(c.locked||[]).length ? ` · edited by you: ${esc(c.locked.join(", "))}` : ""}</small>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px">
      <label style="display:grid;gap:2px;font-size:11px">Type<select id="kb-cat-doc_type-${id}" onfocus="markEditing()">${opts(o.types, c.doc_type)}</select></label>
      ${field("subtype","Kind",c.subtype)}${field("period","Period / version",c.period)}${field("issuer","Issuer",c.issuer)}
      <label style="display:grid;gap:2px;font-size:11px">Status<select id="kb-cat-status-${id}" onfocus="markEditing()">${opts(o.statuses, c.status)}</select></label>
    </div>
    <label style="display:grid;gap:2px;font-size:11px">Summary (optional)<textarea id="kb-cat-summary-${id}" rows="2" onfocus="markEditing()">${esc(c.summary||"")}</textarea></label>
    <div style="display:flex;gap:8px;align-items:center"><button class="save" onclick="kbCatalogSave('${id}')">Save</button>
      <button class="quiet" onclick="kbCatalogEdit(null)">Cancel</button>
      <small style="color:var(--ink3)">Metadata only: nothing is re-embedded or re-OCR'd, and these fields stay yours on re-index.</small></div></div>`;
}

function kbCatalogFilter(key){ kbState.catFilter = key; kbRerender(); }
function kbCatalogEdit(id){ kbState.catEdit = kbState.catEdit === id ? null : id; if (id) markEditing(); else editing = false; kbRerender(); }

async function kbCatalogSave(id){
  const get = name => (document.getElementById(`kb-cat-${name}-${id}`) || {}).value || "";
  const workspace = (D.documents && D.documents.workspace) || "default";
  const fields = {doc_type:get("doc_type"), subtype:get("subtype"), period:get("period"), issuer:get("issuer"), status:get("status"), summary:get("summary")};
  const result = await postJSON("/api/documents/catalog", {document_id:id, workspace, fields});
  kbState.message = result.error || "Catalog entry saved.";
  if (!result.error) kbState.catEdit = null;
  editing = false;
  await refresh();
}

async function kbCatalogRefresh(){
  if (kbState.busy) return;
  const workspace = (D.documents && D.documents.workspace) || "default";
  kbState.busy = true; kbState.message = "Refreshing catalog…"; kbRerender();
  const result = await postJSON("/api/documents/catalog/refresh", {workspace});
  kbState.busy = false;
  kbState.message = result.error || `Catalog refreshed: ${result.documents} documents (${result.catalogued} new, ${result.updated} updated, ${result.unchanged} unchanged). Nothing was re-read, OCR'd or embedded.`;
  editing = false;
  await refresh();
}

function kbDocumentRows(documents){
  if (!documents.length) return `<div class="kb-empty"><b>No documents indexed.</b><span>Choose files or drop a folder to build the first searchable collection.</span></div>`;
  return documents.map(document => `<article class="kb-doc ${kbState.selected===document.document_id?"selected":""}">
    <button class="kb-doc-main" onclick="kbViewChunks('${document.document_id}')">
      <span class="kb-filemark">${esc((document.title||"D").slice(0,1).toUpperCase())}</span>
      <span class="kb-doc-copy"><b>${esc(document.title)}</b><small>${document.chunk_count} chunks · ${esc(document.media_type)} · ${(document.source_uri||"").startsWith("s3://")?"MinIO":"local file"}</small>${kbCatalogLine(document)}</span>
      <span class="kb-ready">${esc(document.status)}</span>
    </button>
    <div class="kb-doc-actions"><button onclick="kbCatalogEdit('${document.document_id}')">Edit</button>
      <button title="Re-reads the original. Unchanged content costs nothing: OCR pages and vectors come from the cache." onclick="kbDocumentAction('reindex','${document.document_id}')">Re-index</button>
      <button class="danger" onclick="kbDocumentAction('delete','${document.document_id}')">Remove</button></div>${kbCatalogEditor(document)}
  </article>`).join("");
}

function kbEvidence(){
  if (kbState.searching) return `<div class="kb-empty"><span class="kb-spinner"></span><span>Searching indexed chunks…</span></div>`;
  if (!kbState.hits.length) return "";
  return `<div class="kb-evidence">${kbState.hits.map(hit => `<article>
    <div><b>${esc(hit.title)}</b><span>#chunk-${hit.ordinal}${hit.page_start ? ` · ${hit.page_end && hit.page_end !== hit.page_start ? `pp.${hit.page_start}-${hit.page_end}` : `p.${hit.page_start}`}` : ""}</span></div>
    <p>${esc(hit.text)}</p><small>${esc(hit.source_uri)}</small></article>`).join("")}</div>`;
}

function kbEvalRows(result){
  if (!result.cases.length) return `<div class="kb-empty"><b>No cases in this question set.</b></div>`;
  return `<table class="kb-eval-table"><thead><tr><th>Lang</th>${result.cases.some(r => r.format)?"<th>Format</th>":""}<th>Query</th><th>Expected</th><th>Got</th><th>Keywords</th><th>Latency</th>${result.use_llm?"<th>Answer</th>":""}</tr></thead><tbody>
    ${result.cases.map(row => `<tr class="${row.hit?"ok":"bad"}">
      <td><span class="kb-lang">${esc(row.lang||"en")}</span></td>
      ${result.cases.some(r => r.format)?`<td>${row.format?`<span class="kb-lang">${esc(row.format)}</span>`:""}</td>`:""}
      <td>${esc(row.query)}</td>
      <td>${esc(row.expected_title)}</td>
      <td class="got">${row.hit?"✓ "+esc(row.expected_title):esc((row.got_titles||[]).join(", ")||"(nothing)")}</td>
      <td>${row.keyword_total?`${row.keyword_hits.length}/${row.keyword_total}`:"—"}</td>
      <td>${row.latency_ms}ms</td>
      ${result.use_llm?`<td>${row.answer_error?`<span class="kb-answer-bad">${esc(row.answer_error)}</span>`:`<details class="kb-answer"><summary>${row.keyword_total?`${(row.answer_keyword_hits||[]).length}/${row.keyword_total} kw`:"answer"} · ${row.llm_ms}ms</summary><p>${esc(row.answer||"(empty)")}</p></details>`}</td>`:""}
    </tr>`).join("")}
  </tbody></table>`;
}

function kbEvalChips(groups){
  const entries = Object.entries(groups || {});
  if (entries.length < 2) return "";
  return `<div class="kb-eval-langs">${entries.map(([name, t]) => `<span class="kb-lang">${esc(name)}</span><b>${Math.round(t.recall*100)}%</b><small>${t.hits}/${t.total}</small>`).join("")}</div>`;
}

function kbEvalRerankNote(result){
  if (!result.reranked) return "";
  const bad = result.rerank_errors
    ? `<b class="kb-answer-bad">${result.rerank_errors} of ${result.total} searches fell back to vector order</b> (${esc(result.rerank_last_error||"rerank failed")}) — these numbers are partly NOT reranked.`
    : "All searches were reranked.";
  return `<div class="kb-run-message">Reranked · ${esc(result.config.rerank.provider)} · ${result.config.rerank.candidates} candidates. ${bad}</div>`;
}

function kbEvalHiddenNote(result){
  const expired = result.expired;
  if (!expired || !expired.cases.length) return "";
  const leak = expired.leaked
    ? `<b class="kb-answer-bad">${expired.leaked} of them still returned the hidden document</b>`
    : "none of them returned it, as intended";
  return `<div class="kb-run-message">${expired.cases.length} question(s) ask about ${expired.documents.map(esc).join(", ")}, marked superseded/excluded in the catalog. They are <b>expired</b>: not scored above, and checked the other way round — ${leak}. Set the status back to "current" to score them normally.</div>`;
}

function kbEvalColdNote(result){
  // One query that loaded a local model into memory can dwarf the rest;
  // say so instead of letting it read as the typical latency.
  if (result.median_latency_ms > 0 && result.first_query_ms > Math.max(500, result.median_latency_ms * 5))
    return `<div class="kb-run-message">First query took ${result.first_query_ms}ms (model cold start) — median excludes its influence, average does not (${result.avg_latency_ms}ms).</div>`;
  return "";
}

function kbEvalSection(data){
  const sets = (data.documents && data.documents.question_sets) || [];
  const result = kbState.evalResult;
  return `<section class="kb-eval"><div class="kb-section-head"><div><span class="kb-kicker">Retrieval eval</span><h3>Score real questions against this workspace</h3></div></div>
    ${sets.length ? `<div class="kb-eval-controls">
      <select onfocus="markEditing()" onchange="kbEvalPick(this)">
        <option value="">Choose a question set…</option>
        ${sets.map(s => `<option value="${esc(s.name)}" ${kbState.evalSet===s.name?"selected":""}>${esc(s.name)} (${s.count})${s.source==="imported"?" · imported":""}</option>`).join("")}
      </select>
      <button onclick="kbRunEval()" ${(!kbState.evalSet||kbState.evalRunning)?"disabled":""}>${kbState.evalRunning?"Running…":"Run eval"}</button>
      <button class="quiet" onclick="kbEvalImportPick()">Import JSON</button>
    </div>
    <label class="kb-eval-llm">Results per question (k) <select onfocus="markEditing()" onchange="kbEvalK(this)">${[3,5,6,10].map(n => `<option value="${n}" ${kbState.evalK===n?"selected":""}>${n}</option>`).join("")}</select> <small>— the agent's search_documents returns 6 by default, so 6 shows what it actually sees</small></label>
    <label class="kb-eval-llm">Workspace <input value="${esc(kbState.evalWorkspace || (D.documents && D.documents.workspace) || "default")}" onfocus="markEditing()" onchange="kbEvalWorkspace(this)" style="width:12em"> <small>— a set only scores against the workspace that holds its documents (synthetic_denso_xl → denso-xl)</small></label>
    ${((data.documents && data.documents.rerank) || {}).configured ? `<label class="kb-eval-llm"><input type="checkbox" onchange="kbEvalRerank(this)" ${kbState.evalRerank!==false?"checked":""}> Rerank the top ${data.documents.rerank.candidates} candidates with ${esc(data.documents.rerank.provider)} <small>${data.documents.rerank.local ? "— runs on this machine, nothing leaves it; if the reranker is not running the search falls back to vector order and the run says so." : `— sends the question and those chunks' text to ${esc(data.documents.rerank.provider)}, one call per question.`} Unchecked = vector order only.</small></label>` : ""}
    <label class="kb-eval-llm"><input type="checkbox" onchange="kbEvalLlm(this)" ${kbState.evalUseLlm?"checked":""}> Also answer with the chat model <small>${esc((D.provider||"")+" · "+(D.model||"chat model"))} — sends the retrieved chunks to it, one call per question</small></label>` : `<div class="kb-empty"><b>No question sets yet.</b><span>Import a JSON/JSONL file — each case needs a query and an expected_title (see evals/documents/denso_qa.jsonl).</span><div class="kb-eval-controls" style="margin-top:10px"><button onclick="kbEvalImportPick()">Import JSON</button></div></div>`}
    <input id="kb-eval-file" hidden type="file" accept=".json,.jsonl,application/json" onchange="kbEvalImport(this)">
    ${kbState.evalNote ? `<div class="kb-run-message">${esc(kbState.evalNote)}</div>` : ""}
    ${kbState.evalError ? `<div class="kb-run-message bad">${esc(kbState.evalError)}</div>` : ""}
    ${result ? `${kbEvalChips(result.by_lang)}${kbEvalChips(result.by_format)}<div class="kb-eval-totals"><div><b>${Math.round(result.recall*100)}%</b><span>recall @${result.k}</span></div><div><b>${result.hits}/${result.total}</b><span>hits</span></div>${(result.answer_in_chunks||{}).total?`<div><b>${Math.round(result.answer_in_chunks.rate*100)}%</b><span>answer in chunks · ${result.answer_in_chunks.found}/${result.answer_in_chunks.total}</span></div>`:""}<div><b>${result.median_latency_ms}ms</b><span>median latency</span></div><div><b>${result.p95_latency_ms}ms</b><span>p95</span></div>${result.use_llm?`<div><b>${result.answer_keyword_rate==null?"—":Math.round(result.answer_keyword_rate*100)+"%"}</b><span>answer keywords</span></div><div><b>${result.llm_median_ms}ms</b><span>median LLM · p95 ${result.llm_p95_ms}ms${result.llm_errors?` · ${result.llm_errors} failed`:""}</span></div>`:""}</div>${kbEvalRerankNote(result)}${kbEvalHiddenNote(result)}${kbEvalColdNote(result)}${result.saved_as?`<div class="kb-run-message">Run saved as .tini/eval_runs/${esc(result.saved_as)}</div>`:""}${kbEvalRows(result)}` : ""}
  </section>`;
}

function kbChunkPreview(documents){
  if (!kbState.selected) return "";
  const document = documents.find(item => item.document_id === kbState.selected);
  if (!document) return "";
  const content = kbState.chunksLoading
    ? `<div class="kb-empty"><span class="kb-spinner"></span><span>Loading chunks…</span></div>`
    : kbState.chunks.map(chunk => `<article class="kb-chunk"><header><b>Chunk ${chunk.ordinal}</b><span>${chunk.embedded?"vector · ":""}${chunk.char_start}–${chunk.char_end}</span></header><p>${esc(chunk.text)}</p></article>`).join("");
  return `<section class="kb-preview"><div class="kb-section-head"><div><span class="kb-kicker">Chunk inspector</span><h3>${esc(document.title)}</h3></div>
    <button class="kb-close" onclick="kbState.selected=null;kbRerender()" aria-label="Close chunk inspector">×</button></div>${content}</section>`;
}

function kbOcrCost(cost){
  if (!cost || !cost.documents) return "";
  const usd = value => "$" + (value >= 0.1 ? value.toFixed(2) : value.toFixed(4));
  const rate = cost.rate_per_million ? ` · est. at $${cost.rate_per_million.in}/$${cost.rate_per_million.out} per M tokens in/out` : "";
  const waste = cost.wasted_cost > 0 ? ` · ${usd(cost.wasted_cost)} spent re-OCRing unchanged files` : "";
  return `<i></i><b title="${esc(`${cost.documents} scanned documents, ${cost.pages} pages, ${usd(cost.total_cost)} total, ${usd(cost.avg_cost_per_page)} per page${rate}${waste}`)}">${usd(cost.avg_cost_per_document)}</b><span>avg OCR / doc</span>`;
}

function documentsView(data){
  const info = data.documents || {items:[], chunk_count:0, workspace:"default", max_file_bytes:0};
  if (info.error){
    return `<div class="kb-hero"><div><span class="kb-kicker">Workspace · ${esc(info.workspace||"default")}</span>
        <p>Turn internal files into evidence the agent can cite.</p></div></div>
      <div class="card" style="border-color:var(--bad)">
        <b style="color:var(--bad)">Documents unavailable</b>
        <p class="meta" style="margin-top:6px">${esc(info.error)}</p>
        <p class="meta">Check the document index / embedding / reranking settings in your .env, then reload.</p>
      </div>`;
  }
  const documents = info.items || [];
  const latest = kbState.recent.slice(0,3).map(item => `<div class="kb-log ${item.ok?"ok":"bad"}"><span></span><b>${esc(item.name)}</b><small>${esc(item.detail)}</small></div>`).join("");
  return `<div class="kb-hero"><div><span class="kb-kicker">Workspace · ${esc(info.workspace)}</span>
      <p>Turn internal files into evidence the agent can cite.</p></div>
    <div class="kb-totals"><b>${documents.length}</b><span>documents</span><i></i><b>${info.chunk_count||0}</b><span>chunks</span><i></i><b>${info.vector_count||0}</b><span>vectors</span>${kbOcrCost(info.ocr_cost)}</div></div>
    <div class="kb-grid"><section class="kb-ingest-panel">
      <div class="kb-section-head"><div><span class="kb-kicker">Ingestion desk</span><h3>Add knowledge</h3></div>
        <span class="kb-limit">max ${kbBytes(info.max_file_bytes)}</span></div>
      <div id="kb-drop" class="kb-drop" tabindex="0" ondragover="kbDrag(event,true)" ondragleave="kbDrag(event,false)" ondrop="kbDrop(event)" onkeydown="if(event.key==='Enter')kbPick('files')">
        <span class="kb-drop-glyph">↧</span><b>Drop documents here</b><span>Text, Markdown, HTML, PDF, DOCX and structured text</span>
        <div><button onclick="event.stopPropagation();kbPick('files')">Choose files</button><button class="quiet" onclick="event.stopPropagation();kbPick('folder')">Choose folder</button></div></div>
      <input id="kb-file-input" hidden type="file" multiple accept=".txt,.md,.markdown,.html,.htm,.csv,.json,.yaml,.yml,.pdf,.docx" onchange="kbPicked(this)">
      <input id="kb-folder-input" hidden type="file" multiple accept=".txt,.md,.markdown,.html,.htm,.csv,.json,.yaml,.yml,.pdf,.docx" webkitdirectory directory onchange="kbPicked(this)">
      <div class="kb-run-head"><span>${kbState.filename?esc(kbState.filename):"Pipeline ready"}</span><small>${kbState.busy?"processing":"local · no model call"}</small></div>
      ${kbFlow()}<div class="kb-run-message ${kbState.failed?"bad":""}">${esc(kbState.message || "Files are hashed, chunked, and committed transactionally.")}</div>${latest}
    </section>
    <section class="kb-library-panel"><div class="kb-section-head"><div><span class="kb-kicker">Document library</span><h3>Indexed sources</h3></div><span class="kb-limit">${documents.length} total</span></div>
      ${kbCatalogBar(documents)}<div class="kb-doc-list">${kbDocumentRows(kbCatalogVisible(documents))}</div></section></div>
    <section class="kb-probe"><div class="kb-section-head"><div><span class="kb-kicker">Retrieval probe</span><h3>Test what the agent can find</h3></div><span class="kb-limit">top 6 chunks</span></div>
      <div class="kb-search"><input id="kb-query" value="${esc(kbState.query)}" placeholder="e.g. What is the incident rollback procedure?" onfocus="markEditing()" onkeydown="kbSearchKey(event)"><button onclick="kbSearch()">Search index</button></div>${kbEvidence()}</section>
    ${kbEvalSection(data)}
    ${kbGenSection(data)}
    ${kbChunkPreview(documents)}`;
}

// --- Generation eval: what the agent does with what it retrieved -------------------
// Runs the real turn (system prompt, search_documents, the loop) over a fixed question
// set and scores answers, citations, reply language, faithfulness and declining. It
// calls a chat model (and optionally a judge model) for every question, so nothing
// starts until Run is pressed and the estimate is confirmed. Results stream in
// question by question and are saved under .tini/generation_runs/.
let kbGen = {
  sources:null, limit:"", arms:"on", genProvider:"", genModel:"",
  judgeOn:false, judgeProvider:"", judgeModel:"",
  plan:null, error:"", running:false, done:0, total:0, cancelling:false, abort:null,
  arms_data:{}, order:[], runs:null, message:"", loaded:"", open:{},
};

function kbGenInfo(){ return (D && D.documents && D.documents.generation) || null; }

function kbGenSources(){
  const info = kbGenInfo();
  if (!info) return [];
  if (!kbGen.sources) kbGen.sources = Object.keys(info.sources);
  return kbGen.sources;
}

function kbGenPayload(){
  const info = kbGenInfo();
  const payload = {sources:kbGenSources(), limit:Number(kbGen.limit) || 0, arms:kbGen.arms, generator:{}, judge:null};
  if (kbGen.genProvider || kbGen.genModel) payload.generator = {provider:kbGen.genProvider || info.generator.provider, model:kbGen.genModel};
  if (kbGen.judgeOn) payload.judge = {provider:kbGen.judgeProvider, model:kbGen.judgeModel};
  return payload;
}

function kbGenSet(field, value){ kbGen[field] = value; kbGen.plan = null; kbGen.error = ""; }
function kbGenToggleSource(name, checked){
  const now = new Set(kbGenSources());
  if (checked) now.add(name); else now.delete(name);
  kbGen.sources = Array.from(now); kbGen.plan = null; kbGenRerender();
}
function kbGenRerender(){ editing = false; kbRerender(); }

async function kbGenRuns(){
  const result = await postJSON("/api/documents/generation/runs", {});
  kbGen.runs = result.runs || [];
}

async function kbGenPreview(){
  kbGen.error = ""; kbGen.plan = null; kbGenRerender();
  const plan = await postJSON("/api/documents/generation/plan", kbGenPayload());
  if (plan.error) kbGen.error = plan.error; else kbGen.plan = plan;
  kbGenRerender();
  return plan;
}

function kbGenEstimateText(plan){
  const e = plan.estimate, n = x => Number(x).toLocaleString();
  const judge = plan.judge ? `, judge ${esc(plan.judge.provider)} · ${esc(plan.judge.model)}` : ", no judge";
  return `${plan.questions} questions × ${plan.arms.length} run${plan.arms.length>1?"s":""} with ${plan.generator.provider} · ${plan.generator.model}${judge}: about ${n(e.generator_calls)} generator calls${e.judge_calls ? ` and ${n(e.judge_calls)} judge calls` : ""}, roughly ${n(Math.round(e.input_tokens/1000))}k input and ${n(Math.round(e.output_tokens/1000))}k output tokens (${e.basis}).`;
}

async function kbGenRun(){
  if (kbGen.running) return;
  const plan = await kbGenPreview();
  if (plan.error) return;
  const text = kbGenEstimateText(plan).replace(/&[a-z]+;/g, "") + (plan.self_judged ? "\n\nThe judge is the same model as the generator; a model grading its own answers flatters itself." : "") + "\n\nThis calls the models and may cost money. Continue?";
  if (!confirm(text)) return;
  kbGen.running = true; kbGen.cancelling = false; kbGen.done = 0; kbGen.total = plan.questions * plan.arms.length;
  kbGen.arms_data = {}; kbGen.order = plan.arms; kbGen.error = ""; kbGen.loaded = "";
  plan.arms.forEach(a => { kbGen.arms_data[a] = {rows:[], summary:null}; });
  kbGen.abort = new AbortController(); kbGenRerender();
  try {
    const response = await fetch("/api/documents/generation/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(kbGenPayload()), signal:kbGen.abort.signal});
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "", final = null;
    for (;;){
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      let split;
      while ((split = buffer.indexOf("\n\n")) >= 0){
        const line = buffer.slice(0, split); buffer = buffer.slice(split + 2);
        if (!line.startsWith("data:")) continue;
        let event; try { event = JSON.parse(line.slice(5).trim()); } catch(e){ continue; }
        if (event.kind === "row"){ kbGen.arms_data[event.arm].rows.push(event.row); kbGen.done = event.done; kbGenRerender(); }
        else if (event.kind === "arm_done"){ kbGen.arms_data[event.arm].summary = event.summary; kbGenRerender(); }
        else if (event.kind === "done") final = event;
      }
    }
    if (final && final.error) kbGen.error = final.error;
    else if (final) kbGen.message = final.cancelled ? "Cancelled — the questions already answered are kept below." : `Done. Saved as .tini/generation_runs/${final.saved_as || "(not saved)"}`;
  } catch(error){
    kbGen.message = error && error.name === "AbortError" ? "Cancelled — the questions already answered are kept below." : "";
    if (!(error && error.name === "AbortError")) kbGen.error = String(error.message || error);
  }
  kbGen.running = false; kbGen.abort = null;
  await kbGenRuns(); kbGenRerender();
}

function kbGenCancel(){ if (kbGen.abort){ kbGen.cancelling = true; kbGen.abort.abort(); } }

async function kbGenLoad(name){
  const result = await postJSON("/api/documents/generation/load", {name});
  if (result.error){ kbGen.error = result.error; kbGenRerender(); return; }
  kbGen.arms_data = {}; kbGen.order = Object.keys(result.arms || {});
  kbGen.order.forEach(a => { kbGen.arms_data[a] = result.arms[a]; });
  kbGen.loaded = name; kbGen.error = ""; kbGen.message = `Loaded ${name} · ${result.generator ? result.generator.provider + " · " + result.generator.model : ""}${result.judge ? " · judge " + result.judge.model : " · no judge"}${result.cancelled ? " · cancelled early" : ""}`;
  kbGenRerender();
}

function kbGenFrac(a, b){ return b ? `${a}/${b}` : "—"; }

function kbGenSummaryTable(summary){
  if (!summary) return `<div class="kb-run-message">Waiting for the first question…</div>`;
  const head = ["group","answered","retrieval-limited","generation missed","cited","page","reply language","faithful","declined","invented"];
  const rows = Object.entries(summary).filter(([, g]) => g && typeof g === "object").map(([name, g]) => `<tr>
    <td>${esc(name)}</td><td>${kbGenFrac(g.answered, g.answerable)}</td><td>${g.retrieval_limited}</td>
    <td>${kbGenFrac(g.generation_missed, g.generation_of_evidence)}</td><td>${kbGenFrac(g.cited, g.answered_and_cited_base)}</td><td>${kbGenFrac(g.page_ok, g.page_base)}</td>
    <td>${kbGenFrac(g.language_ok, g.n)}</td><td>${kbGenFrac(g.faithful, g.judged)}</td>
    <td>${kbGenFrac(g.declined, g.unanswerable)}</td><td>${kbGenFrac(g.invented, g.unanswerable)}</td></tr>`).join("");
  return `<table class="kb-eval-table"><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>
    <small style="color:var(--ink3)">answered: every expected keyword is in the reply · retrieval-limited: the chunks the agent saw did not contain it · generation missed: they did, the reply does not · page: the reply names a page that holds the answer (PDF answers only) · faithful / declined / invented come from the judge model. Counts, not percentages.</small>`;
}

function kbGenMark(ok){ return ok ? "✓" : `<b class="kb-answer-bad">✗</b>`; }

// The page column: only an answered PDF question has one (expected pages come from the chunk that held the answer).
function kbGenPage(r){
  if (r.abstain || !r.pages_expected || !r.reply_has_answer) return "—";
  const shown = (r.pages_cited || []).length ? `p.${r.pages_cited.join(",")}` : "none";
  return `<span title="the answer is on p.${esc(r.pages_expected.join(","))}">${kbGenMark(r.page_ok)} <small>${esc(shown)}</small></span>`;
}

function kbGenRows(arm, rows){
  if (!rows.length) return "";
  return `<table class="kb-eval-table"><thead><tr><th>Source</th><th>Lang</th><th>Question</th><th>Answer</th><th>Cited</th><th>Page</th><th>Language</th><th>Judge</th><th>Time</th></tr></thead><tbody>
    ${rows.map((r, i) => {
      if (r.error) return `<tr class="bad"><td>${esc(r.source)}</td><td><span class="kb-lang">${esc(r.lang)}</span></td><td>${esc(r.query)}</td><td colspan="6"><span class="kb-answer-bad">${esc(r.error)}</span></td></tr>`;
      const j = r.judge, key = `${arm}:${i}`;
      const answer = r.abstain ? "—" : (r.reply_has_answer ? "✓" : (r.evidence_has_answer ? `<b class="kb-answer-bad">✗ model</b>` : `<b class="kb-answer-bad">✗ retrieval</b>`));
      const verdict = !j ? "—" : (r.abstain ? (j.abstained ? (j.supported==="yes" ? "declined ✓" : "declined, partly unsupported") : `<b class="kb-answer-bad">invented</b>`) : `faithful: ${esc(j.supported)}`);
      return `<tr class="${(r.abstain ? (j && j.abstained) : r.reply_has_answer) ? "ok" : "bad"}"><td>${esc(r.source)}</td><td><span class="kb-lang">${esc(r.lang)}</span></td>
        <td>${esc(r.query)}<details class="kb-answer"><summary>reply · ${(r.searches||[]).length} search${(r.searches||[]).length===1?"":"es"}</summary><p>${esc(r.reply || "(empty)")}</p>${j && (j.unsupported||[]).length ? `<p><b>Unsupported by the evidence (judge):</b> ${j.unsupported.map(esc).join(" · ")}</p>` : ""}${(r.searches||[]).length ? `<p><small>Searched: ${(r.searches||[]).map(esc).join(" · ")}</small></p>` : ""}</details></td>
        <td>${answer}</td><td>${r.abstain ? "—" : kbGenMark(r.cited)}</td><td>${kbGenPage(r)}</td><td>${kbGenMark(r.language_ok)}</td><td>${verdict}</td><td>${r.seconds}s</td></tr>`;
    }).join("")}</tbody></table>`;
}

function kbGenCompare(){
  if (kbGen.order.length < 2) return "";
  const [a, b] = kbGen.order.map(k => kbGen.arms_data[k].rows.filter(r => !r.error));
  const byId = Object.fromEntries(b.map(r => [r.id, r]));
  const pairs = a.filter(r => byId[r.id]);
  const diff = key => ({fixed: pairs.filter(r => r[key] && !byId[r.id][key]).length, broke: pairs.filter(r => !r[key] && byId[r.id][key]).length});
  const lang = diff("language_ok"), ans = diff("reply_has_answer");
  return `<div class="kb-run-message">Language line <b>on</b> vs <b>off</b>, same questions: reply language correct on ${lang.fixed} more with it and ${lang.broke} fewer; answered ${ans.fixed} more and ${ans.broke} fewer (${pairs.length} paired questions). Small differences are noise at this size.</div>`;
}

function kbGenSection(data){
  const info = kbGenInfo();
  if (!info) return "";
  if (kbGen.runs === null && !kbGen.runsRequested){ kbGen.runsRequested = true; kbGenRuns().then(kbGenRerender); }
  const sources = new Set(kbGenSources());
  const arms = kbGen.order;
  const progress = kbGen.running ? `<div class="kb-run-message"><b>${kbGen.done}/${kbGen.total}</b> questions${kbGen.cancelling ? " · cancelling…" : ""} <progress max="${kbGen.total}" value="${kbGen.done}" style="width:200px;vertical-align:middle"></progress></div>` : "";
  const provOptions = (selected, onlyConfigured) => info.providers.filter(p => !onlyConfigured || p.configured).map(p => `<option value="${esc(p.id)}" ${selected===p.id?"selected":""}>${esc(p.id)}${p.configured?"":" (no key)"}</option>`).join("");
  const runs = kbGen.runs;
  return `<section class="kb-eval"><div class="kb-section-head"><div><span class="kb-kicker">Generation eval</span><h3>Score what the agent does with what it retrieves</h3></div></div>
    <p class="meta" style="margin:0 0 8px">Runs the real turn (system prompt, <code>search_documents</code>, the loop) over ${info.total} fixed questions in en/ja/vi and checks the answer, the citation, the reply language, whether it stays faithful to the evidence, and whether it declines when the documents do not hold the answer. <b>Every question calls the model${kbGen.judgeOn ? " and the judge" : ""}: it costs money and minutes, and nothing starts until you confirm.</b> It uses no memory and writes no chat log.</p>
    <div class="kb-eval-controls" style="flex-wrap:wrap;gap:8px 14px">
      ${Object.entries(info.sources).map(([name, count]) => `<label><input type="checkbox" ${sources.has(name)?"checked":""} onchange="kbGenToggleSource('${name}', this.checked)"> ${esc(name)} (${count})</label>`).join("")}
      <label>Questions <input type="number" min="0" max="${info.max_questions}" placeholder="all" value="${esc(kbGen.limit)}" style="width:5em" onfocus="markEditing()" onchange="kbGenSet('limit', this.value)"> <small>spread over all sources</small></label>
    </div>
    <div class="kb-eval-controls" style="flex-wrap:wrap;gap:8px 14px">
      <label>Generator <select onfocus="markEditing()" onchange="kbGenSet('genProvider', this.value)"><option value="">current · ${esc(info.generator.provider)}</option>${provOptions(kbGen.genProvider, true)}</select>
        <input placeholder="${esc(info.generator.model)}" value="${esc(kbGen.genModel)}" style="width:14em" onfocus="markEditing()" onchange="kbGenSet('genModel', this.value)"></label>
      <label><input type="checkbox" ${kbGen.judgeOn?"checked":""} onchange="kbGenSet('judgeOn', this.checked); kbGenRerender()"> Judge with a second model</label>
      ${kbGen.judgeOn ? `<label>Judge <select onfocus="markEditing()" onchange="kbGenSet('judgeProvider', this.value)"><option value="">provider…</option>${provOptions(kbGen.judgeProvider, true)}</select>
        <input placeholder="model id" value="${esc(kbGen.judgeModel)}" style="width:14em" onfocus="markEditing()" onchange="kbGenSet('judgeModel', this.value)"></label>` : `<small>No judge: only the checks that need no model (answer, citation, language). Faithfulness and declining need a judge.</small>`}
      <label>Reply-language line <select onfocus="markEditing()" onchange="kbGenSet('arms', this.value)"><option value="on" ${kbGen.arms==="on"?"selected":""}>on (as shipped)</option><option value="off" ${kbGen.arms==="off"?"selected":""}>off</option><option value="both" ${kbGen.arms==="both"?"selected":""}>both (A/B, runs twice)</option></select></label>
    </div>
    <div class="kb-eval-controls">
      <button onclick="kbGenPreview()" class="quiet" ${kbGen.running?"disabled":""}>Preview cost</button>
      <button onclick="kbGenRun()" ${kbGen.running?"disabled":""}>${kbGen.running?"Running…":"Run generation eval"}</button>
      ${kbGen.running ? `<button class="danger" onclick="kbGenCancel()">Cancel</button>` : ""}
    </div>
    ${kbGen.plan ? `<div class="kb-run-message">${kbGenEstimateText(kbGen.plan)}${kbGen.plan.self_judged ? " <b>The judge is the generator: it will flatter itself.</b>" : ""}</div>` : ""}
    ${kbGen.error ? `<div class="kb-run-message bad">${esc(kbGen.error)}</div>` : ""}
    ${kbGen.message ? `<div class="kb-run-message">${esc(kbGen.message)}</div>` : ""}
    ${progress}
    ${arms.map(a => `<h4 style="margin:12px 0 4px">Reply-language line ${a === "on" ? "on" : "off"}</h4>${kbGenSummaryTable(kbGen.arms_data[a].summary)}${kbGenRows(a, kbGen.arms_data[a].rows)}`).join("")}
    ${kbGenCompare()}
    <div class="kb-section-head" style="margin-top:14px"><div><span class="kb-kicker">Past runs</span></div><button class="quiet" onclick="kbGenRuns().then(kbGenRerender)">${runs===null?"Load list":"Refresh"}</button></div>
    ${runs && !runs.length ? `<small>No saved runs yet.</small>` : ""}
    ${(runs||[]).map(r => `<div style="display:flex;gap:10px;align-items:center;padding:3px 0"><button class="quiet" onclick="kbGenLoad('${esc(r.name)}')">${esc(r.created || r.name)}</button>
      <small>${r.generator ? esc(r.generator.provider + " · " + r.generator.model) : ""}${r.judge ? " · judge " + esc(r.judge.model) : " · no judge"}${r.cancelled ? " · cancelled" : ""} — ${Object.entries(r.arms).map(([arm, s]) => `line ${arm}: answered ${esc(s.answered)}, language ${esc(s.language_ok)}, declined ${esc(s.declined)}`).join(" · ")}</small></div>`).join("")}
  </section>`;
}

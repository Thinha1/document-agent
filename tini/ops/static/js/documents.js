// Knowledge Base — browser uploads, real ingestion stages, library and retrieval probe.

const KB_STAGE_ORDER = ["upload", "discover", "parse", "chunk", "embed", "index", "ready"];
const KB_EVENT_STAGE = {
  uploaded:"discover", discovered:"parse", loading:"parse", chunking:"chunk",
  embedding:"embed", indexing:"index", created:"ready", updated:"ready", skipped:"ready",
};
let kbState = {
  busy:false, stage:"", failed:false, filename:"", message:"",
  recent:[], selected:null, chunks:[], chunksLoading:false,
  query:"", hits:[], searching:false,
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
  if (action === "delete" && !confirm(`Remove “${title}” from the index? The source file will stay on disk.`)) return;
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

function kbDocumentRows(documents){
  if (!documents.length) return `<div class="kb-empty"><b>No documents indexed.</b><span>Choose files or drop a folder to build the first searchable collection.</span></div>`;
  return documents.map(document => `<article class="kb-doc ${kbState.selected===document.document_id?"selected":""}">
    <button class="kb-doc-main" onclick="kbViewChunks('${document.document_id}')">
      <span class="kb-filemark">${esc((document.title||"D").slice(0,1).toUpperCase())}</span>
      <span class="kb-doc-copy"><b>${esc(document.title)}</b><small>${document.chunk_count} chunks · ${esc(document.media_type)}</small></span>
      <span class="kb-ready">${esc(document.status)}</span>
    </button>
    <div class="kb-doc-actions"><button onclick="kbDocumentAction('reindex','${document.document_id}')">Re-index</button>
      <button class="danger" onclick="kbDocumentAction('delete','${document.document_id}')">Remove</button></div>
  </article>`).join("");
}

function kbEvidence(){
  if (kbState.searching) return `<div class="kb-empty"><span class="kb-spinner"></span><span>Searching indexed chunks…</span></div>`;
  if (!kbState.hits.length) return "";
  return `<div class="kb-evidence">${kbState.hits.map(hit => `<article>
    <div><b>${esc(hit.title)}</b><span>#chunk-${hit.ordinal}</span></div>
    <p>${esc(hit.text)}</p><small>${esc(hit.source_uri)}</small></article>`).join("")}</div>`;
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

function documentsView(data){
  const info = data.documents || {items:[], chunk_count:0, workspace:"default", max_file_bytes:0};
  const documents = info.items || [];
  const latest = kbState.recent.slice(0,3).map(item => `<div class="kb-log ${item.ok?"ok":"bad"}"><span></span><b>${esc(item.name)}</b><small>${esc(item.detail)}</small></div>`).join("");
  return `<div class="kb-hero"><div><span class="kb-kicker">Workspace · ${esc(info.workspace)}</span>
      <p>Turn internal files into evidence the agent can cite.</p></div>
    <div class="kb-totals"><b>${documents.length}</b><span>documents</span><i></i><b>${info.chunk_count||0}</b><span>chunks</span><i></i><b>${info.vector_count||0}</b><span>vectors</span></div></div>
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
      <div class="kb-doc-list">${kbDocumentRows(documents)}</div></section></div>
    <section class="kb-probe"><div class="kb-section-head"><div><span class="kb-kicker">Retrieval probe</span><h3>Test what the agent can find</h3></div><span class="kb-limit">top 6 chunks</span></div>
      <div class="kb-search"><input id="kb-query" value="${esc(kbState.query)}" placeholder="e.g. What is the incident rollback procedure?" onfocus="markEditing()" onkeydown="kbSearchKey(event)"><button onclick="kbSearch()">Search index</button></div>${kbEvidence()}</section>
    ${kbChunkPreview(documents)}`;
}

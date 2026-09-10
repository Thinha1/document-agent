// Data-driven graph visualization for the triage workflow.

function graphLayout(wf){
  const names = ["START", ...wf.nodes.map(n => n.name), "END"];
  const layer = {START: 0};
  for (let pass = 0; pass < names.length; pass++)
    wf.edges.forEach(e => {
      const src = layer[e.src] ?? 0;
      layer[e.dst] = Math.max(layer[e.dst] ?? 0, src + 1);
    });
  const cols = [];
  names.forEach(n => {
    if (layer[n] == null) layer[n] = 0;
    (cols[layer[n]] = cols[layer[n]] || []).push(n);
  });
  return {cols: cols.filter(Boolean)};
}

function graphSVG(wf){
  const {cols} = graphLayout(wf);
  const kinds = Object.fromEntries(wf.nodes.map(n => [n.name, n.kind]));
  const W = 168, H = 52, GX = 74, GY = 22, PAD = 14;
  const height = Math.max(...cols.map(c => c.length)) * (H + GY) - GY + PAD * 2;
  const width = cols.length * (W + GX) - GX + PAD * 2;
  const pos = {};
  cols.forEach((col, ci) => col.forEach((name, ri) => {
    const colH = col.length * (H + GY) - GY;
    pos[name] = {x: PAD + ci * (W + GX), y: PAD + (height - PAD * 2 - colH) / 2 + ri * (H + GY)};
  }));
  const nodeId = name => `g-${wf.name}-${name}`;
  const labels = {llm: "one model call", agent: "THE loop, as a node", tool: "code, no model", fn: ""};
  const nodeBox = name => {
    const p = pos[name];
    if (name === "START" || name === "END")
      return `<g class="node" data-node="${nodeId(name)}">
        <rect class="bx" x="${p.x + W/2 - 34}" y="${p.y + H/2 - 15}" width="68" height="30" rx="15"/>
        <text class="nt" x="${p.x + W/2}" y="${p.y + H/2 + 5}" text-anchor="middle" style="font-size:12px">${name}</text></g>`;
    const sub = labels[kinds[name]] || "";
    return `<g class="node" data-node="${nodeId(name)}">
      <rect class="bx" x="${p.x}" y="${p.y}" width="${W}" height="${H}" rx="9"/>
      <text class="nt" x="${p.x + 12}" y="${p.y + 22}">${esc(name)}</text>
      ${sub ? `<text class="ns" x="${p.x + 12}" y="${p.y + 39}">${sub}</text>` : ""}</g>`;
  };
  const edgeLine = e => {
    const a = pos[e.src], b = pos[e.dst];
    const x1 = a.x + (e.src === "START" ? W/2 + 34 : W), y1 = a.y + H/2;
    const x2 = b.x + (e.dst === "END" ? W/2 - 34 : 0), y2 = b.y + H/2;
    const mx = (x1 + x2) / 2;
    return `<path class="flow${e.conditional ? " dash" : ""}" data-edge="g-${wf.name}-${e.src}-${e.dst}"
      d="M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}" marker-end="url(#garr)"/>`;
  };
  return `<div style="overflow-x:auto"><svg viewBox="0 0 ${width} ${height}" class="arch graphchart"
      style="max-width:${width}px" role="img">
    <defs><marker id="garr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
      orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="head"/></marker></defs>
    ${wf.edges.map(edgeLine).join("")}
    ${["START", ...wf.nodes.map(n => n.name), "END"].map(nodeBox).join("")}
  </svg></div>`;
}

let GRAPH_LIVE = null;
let GRAPH_SHOWN = null;

function graphPanel(d){
  const g = d.graph || {enabled:false, workflows:[], runs:[], stats:{quick:0,full:0}};
  const last = (g.runs || [])[0];
  const showing = GRAPH_LIVE || GRAPH_SHOWN || (last || {}).workflow;
  const wf = (g.workflows || []).find(w => w && w.name === showing) || (g.workflows || [])[0];
  const total = g.stats.quick + g.stats.full;
  const split = !total ? `<div class="meta" style="margin:6px 0 10px">no graph turns yet</div>`
    : `<div class="splitbar"><div class="seg-skip" style="width:${Math.round(g.stats.quick/total*100)}%"></div>
       <div class="seg-ret" style="width:${Math.round(g.stats.full/total*100)}%"></div></div>
       <div class="meta" style="margin:6px 0 10px">${g.stats.quick} quick · ${g.stats.full} full</div>`;
  if (!g.enabled && !last)
    return `<div class="card"><div class="meta">Triage routing is off. Enable graph workflows in
      <a class="reveal" onclick="location.hash='settings'">Behaviour</a> to classify each turn first.</div></div>`;
  const status = GRAPH_LIVE ? `<span class="live-dot"></span><b>${esc(GRAPH_LIVE)}</b> running`
    : last ? `last run: <b>${esc(last.workflow || "")}</b>${last.ms ? ` · ${(last.ms/1000).toFixed(1)}s` : ""}`
    : "waiting for a turn";
  return `<div class="card" style="cursor:pointer" onclick="location.hash='graph'">
    ${g.enabled ? split : ""}${wf ? graphSVG(wf) : ""}
    <div class="meta" style="margin-top:8px">${status}</div></div>`;
}

const GRAPH_KINDS = new Set(["graph_start", "node_start", "node_end", "route", "graph_end"]);
function animateGraphStage(ev){
  if (!document.querySelector(".graphchart")) return;
  const workflow = ev.workflow || "";
  if (ev.type === "graph_start"){
    graphLive(workflow);
    hot(`[data-node="g-${workflow}-START"]`, "hot", 1000);
  } else if (ev.type === "node_start"){
    document.querySelectorAll(`[data-node="g-${workflow}-${ev.node}"]`).forEach(el => el.classList.add("hot"));
  } else if (ev.type === "node_end"){
    document.querySelectorAll(`[data-node="g-${workflow}-${ev.node}"]`).forEach(el => el.classList.remove("hot"));
    hot(`[data-node="g-${workflow}-${ev.node}"]`, "done", 900);
  } else if (ev.type === "route"){
    hot(`[data-edge="g-${workflow}-${ev.router}-${ev.target}"]`, "live", 1400);
    hot(`[data-node="g-${workflow}-${ev.target}"]`, "hot", 1400);
  } else if (ev.type === "graph_end"){
    hot(`[data-node="g-${workflow}-END"]`, "hot", 1000);
    graphLive(null);
  }
}

function graphLive(name){
  if (GRAPH_LIVE === name) return;
  if (name) GRAPH_SHOWN = name;
  GRAPH_LIVE = name;
  if (typeof render === "function") render();
}

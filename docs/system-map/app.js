const state = {
  data: null,
  filters: {
    implementedOnly: false,
    showRoadmap: true,
    showStorage: true,
    showLlm: true,
  },
  selectedKey: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheDom();
  bindFilters();
  await loadData();
  window.addEventListener("resize", debounce(drawEdges, 60));
});

function cacheDom() {
  dom.title = document.getElementById("page-title");
  dom.kicker = document.getElementById("page-kicker");
  dom.subtitle = document.getElementById("page-subtitle");
  dom.summary = document.getElementById("page-summary");
  dom.snapshotDate = document.getElementById("snapshot-date");
  dom.legend = document.getElementById("legend");
  dom.facts = document.getElementById("facts-grid");
  dom.graphLayers = document.getElementById("graph-layers");
  dom.graphEdges = document.getElementById("graph-edges");
  dom.pipelineFlow = document.getElementById("pipeline-flow");
  dom.dataModels = document.getElementById("data-models");
  dom.storageMap = document.getElementById("storage-map");
  dom.evidenceList = document.getElementById("evidence-list");
  dom.notes = document.getElementById("architect-notes");
  dom.detailTitle = document.getElementById("detail-title");
  dom.detailStatus = document.getElementById("detail-status");
  dom.detailSummary = document.getElementById("detail-summary");
  dom.detailAttributes = document.getElementById("detail-attributes");
  dom.detailRelations = document.getElementById("detail-relations");
  dom.detailSources = document.getElementById("detail-sources");
}

function bindFilters() {
  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.classList.toggle("is-active", state.filters[button.dataset.filter]);
    button.addEventListener("click", () => {
      const key = button.dataset.filter;
      state.filters[key] = !state.filters[key];
      button.classList.toggle("is-active", state.filters[key]);
      render();
    });
  });
}

async function loadData() {
  try {
    const response = await fetch("./data/graph.json");
    if (!response.ok) {
      throw new Error(`Failed to load graph data (${response.status})`);
    }
    state.data = await response.json();
    seedDetailPanel();
    render();
  } catch (error) {
    dom.summary.textContent = "Unable to load graph data. Serve this directory through a static web server.";
    dom.graphLayers.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function render() {
  if (!state.data) {
    return;
  }
  renderHeader();
  renderLegend();
  renderFacts();
  renderGraph();
  renderPipeline();
  renderDataModels();
  renderStorage();
  renderEvidence();
  renderNotes();
  syncSelectionStyles();
  drawEdges();
}

function renderHeader() {
  const { meta } = state.data;
  dom.title.textContent = meta.title;
  dom.kicker.textContent = meta.kicker;
  dom.subtitle.textContent = meta.subtitle;
  dom.summary.textContent = meta.summary;
  dom.snapshotDate.textContent = meta.snapshot_date;
}

function renderLegend() {
  dom.legend.innerHTML = "";
  state.data.meta.legend.statuses.forEach((entry) => {
    const chip = document.createElement("span");
    chip.className = "legend-chip";
    chip.textContent = `${entry.label}: ${entry.meaning}`;
    dom.legend.appendChild(chip);
  });
}

function renderFacts() {
  dom.facts.innerHTML = "";
  const nodes = filteredNodes();
  const plannedCount = nodes.filter((node) => node.status === "planned").length;
  const implementedCount = nodes.filter((node) => node.status === "implemented").length;
  const partialCount = nodes.filter((node) => node.status === "partial").length;

  const facts = [
    { label: "Visible nodes", value: String(nodes.length) },
    { label: "Implemented", value: String(implementedCount) },
    { label: "Partial", value: String(partialCount) },
    { label: "Planned", value: String(plannedCount) },
  ];

  facts.forEach((fact) => {
    const chip = document.createElement("div");
    chip.className = "fact-chip";
    chip.innerHTML = `<strong>${escapeHtml(fact.value)}</strong> ${escapeHtml(fact.label)}`;
    dom.facts.appendChild(chip);
  });
}

function renderGraph() {
  const nodes = filteredNodes();
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  dom.graphLayers.innerHTML = "";

  const layers = state.data.meta.layers.map((layer) => ({
    ...layer,
    nodes: nodes.filter((node) => node.layer === layer.id),
  })).filter((layer) => layer.nodes.length > 0);

  if (!layers.length) {
    dom.graphLayers.appendChild(emptyState());
    return;
  }

  layers.forEach((layer) => {
    const row = document.createElement("section");
    row.className = "layer-row";

    const label = document.createElement("div");
    label.className = "layer-row__label";
    label.innerHTML = `
      <div class="eyebrow">${escapeHtml(layer.id)}</div>
      <h3>${escapeHtml(layer.label)}</h3>
      <p>${escapeHtml(layer.description)}</p>
    `;

    const nodeGrid = document.createElement("div");
    nodeGrid.className = "layer-row__nodes";

    layer.nodes.forEach((node) => {
      const card = createCardButton({
        key: `node:${node.id}`,
        title: node.label,
        summary: node.summary,
        status: node.status,
        meta: [node.kind, ...node.tags.slice(0, 2)],
        tags: node.tags,
        className: "node-card",
      });
      card.dataset.nodeId = node.id;
      card.addEventListener("click", () => showNodeDetail(node.id));
      nodeGrid.appendChild(card);
    });

    row.append(label, nodeGrid);
    dom.graphLayers.appendChild(row);
  });

  state.visibleEdgeIds = filteredEdges(nodeMap).map((edge) => edge.id);
}

function renderPipeline() {
  dom.pipelineFlow.innerHTML = "";
  const pipeline = state.data.pipelines[0];
  const stages = pipeline.stages.filter((stage) => {
    if (!state.filters.showRoadmap && stage.status === "planned") {
      return false;
    }
    if (state.filters.implementedOnly && stage.status === "planned") {
      return false;
    }
    if (!state.filters.showLlm && stage.llm_config_key) {
      return false;
    }
    return true;
  });

  if (!stages.length) {
    dom.pipelineFlow.appendChild(emptyState());
    return;
  }

  stages.forEach((stage, index) => {
    const card = createCardButton({
      key: `pipeline:${stage.id}`,
      title: stage.label,
      summary: stage.summary,
      status: stage.status,
      meta: stage.outputs.slice(0, 2),
      className: "pipeline-card",
    });

    const indexLabel = document.createElement("div");
    indexLabel.className = "pipeline-card__index";
    indexLabel.textContent = `Stage ${String(index + 1).padStart(2, "0")}`;

    const io = document.createElement("div");
    io.className = "pipeline-io";
    io.innerHTML = `
      ${stage.inputs.length ? `<div><strong>Inputs</strong><ul>${stage.inputs.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
      ${stage.outputs.length ? `<div><strong>Outputs</strong><ul>${stage.outputs.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
      ${stage.writes.length ? `<div><strong>Writes</strong><ul>${stage.writes.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
    `;

    card.prepend(indexLabel);
    card.appendChild(io);
    card.addEventListener("click", () => showPipelineDetail(stage.id));
    dom.pipelineFlow.appendChild(card);
  });
}

function renderDataModels() {
  dom.dataModels.innerHTML = "";
  const models = state.data.data_models.filter((model) => {
    if (state.filters.implementedOnly && model.status === "planned") {
      return false;
    }
    return true;
  });

  if (!models.length) {
    dom.dataModels.appendChild(emptyState());
    return;
  }

  models.forEach((model) => {
    const card = createCardButton({
      key: `model:${model.id}`,
      title: model.label,
      summary: model.summary,
      status: model.status,
      meta: model.fields.slice(0, 3).map((field) => field.name),
      className: "model-card",
    });
    const detail = document.createElement("p");
    detail.className = "card-detail";
    detail.textContent = model.fields.map((field) => `${field.name}: ${field.type}`).join(" | ");
    card.appendChild(detail);
    card.addEventListener("click", () => showModelDetail(model.id));
    dom.dataModels.appendChild(card);
  });
}

function renderStorage() {
  dom.storageMap.innerHTML = "";

  if (!state.filters.showStorage) {
    dom.storageMap.appendChild(emptyState("Storage is hidden by the current filter."));
    return;
  }

  const entries = state.data.storage_map.filter((entry) => {
    if (state.filters.implementedOnly && entry.status === "planned") {
      return false;
    }
    return true;
  });

  if (!entries.length) {
    dom.storageMap.appendChild(emptyState());
    return;
  }

  entries.forEach((entry) => {
    const card = createCardButton({
      key: `storage:${entry.id}`,
      title: entry.label,
      summary: entry.summary,
      status: entry.status,
      meta: [entry.kind, entry.path],
      className: "storage-card",
    });
    const path = document.createElement("p");
    path.className = "storage-card__path";
    path.textContent = entry.path;

    const detail = document.createElement("p");
    detail.className = "storage-card__meta";
    detail.textContent = `Reads: ${entry.reads_by.join(", ") || "none"} | Writes: ${entry.writes_by.join(", ") || "none"}`;

    card.append(path, detail);
    card.addEventListener("click", () => showStorageDetail(entry.id));
    dom.storageMap.appendChild(card);
  });
}

function renderEvidence() {
  dom.evidenceList.innerHTML = "";
  const sourceIds = collectVisibleSourceIds();
  const refs = sourceIds
    .map((id) => state.data.source_refs[id])
    .filter(Boolean);

  if (!refs.length) {
    dom.evidenceList.appendChild(emptyState());
    return;
  }

  refs.forEach((ref) => {
    const card = createCardButton({
      key: `source:${ref.id}`,
      title: ref.label,
      summary: ref.summary,
      status: ref.status || "implemented",
      meta: [ref.path],
      className: "evidence-card",
    });
    const path = document.createElement("p");
    path.textContent = ref.path;
    card.appendChild(path);
    card.addEventListener("click", () => showSourceDetail(ref.id));
    dom.evidenceList.appendChild(card);
  });
}

function renderNotes() {
  dom.notes.innerHTML = "";
  const notes = state.data.architect_notes.filter((note) => {
    if (!state.filters.showRoadmap && note.status === "planned") {
      return false;
    }
    if (state.filters.implementedOnly && note.status === "planned") {
      return false;
    }
    return true;
  });

  if (!notes.length) {
    dom.notes.appendChild(emptyState());
    return;
  }

  notes.forEach((note) => {
    const card = createCardButton({
      key: `note:${note.id}`,
      title: note.label,
      summary: note.summary,
      status: note.status,
      meta: note.tags,
      className: "note-card",
    });
    note.points.forEach((point) => {
      const item = document.createElement("p");
      item.className = "card-detail";
      item.textContent = `- ${point}`;
      card.appendChild(item);
    });
    card.addEventListener("click", () => showNoteDetail(note.id));
    dom.notes.appendChild(card);
  });
}

function drawEdges() {
  if (!state.data || !dom.graphLayers.children.length) {
    return;
  }

  const svg = dom.graphEdges;
  svg.innerHTML = "";
  const rect = dom.graphLayers.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);

  const nodeMap = new Map(filteredNodes().map((node) => [node.id, node]));
  const edges = filteredEdges(nodeMap);

  edges.forEach((edge) => {
    const from = dom.graphLayers.querySelector(`[data-node-id="${edge.from}"]`);
    const to = dom.graphLayers.querySelector(`[data-node-id="${edge.to}"]`);
    if (!from || !to) {
      return;
    }

    const fromRect = from.getBoundingClientRect();
    const toRect = to.getBoundingClientRect();
    const x1 = fromRect.left + fromRect.width / 2 - rect.left;
    const y1 = fromRect.top + fromRect.height / 2 - rect.top;
    const x2 = toRect.left + toRect.width / 2 - rect.left;
    const y2 = toRect.top + toRect.height / 2 - rect.top;
    const mid = Math.abs(x2 - x1) * 0.42 + 40;
    const d = `M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}`;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("class", "edge-path");
    path.dataset.status = edge.status;
    path.dataset.kind = edge.kind;
    svg.appendChild(path);
  });
}

function filteredNodes() {
  return state.data.nodes.filter((node) => {
    if (!state.filters.showRoadmap && node.kind === "roadmap_component") {
      return false;
    }
    if (state.filters.implementedOnly && node.status === "planned") {
      return false;
    }
    if (!state.filters.showLlm && node.tags.includes("llm-boundary")) {
      return false;
    }
    if (!state.filters.showStorage && node.tags.includes("storage")) {
      return false;
    }
    return true;
  });
}

function filteredEdges(nodeMap) {
  return state.data.edges.filter((edge) => {
    if (!nodeMap.has(edge.from) || !nodeMap.has(edge.to)) {
      return false;
    }
    if (!state.filters.showRoadmap && edge.status === "planned") {
      return false;
    }
    if (state.filters.implementedOnly && edge.status === "planned") {
      return false;
    }
    return true;
  });
}

function collectVisibleSourceIds() {
  const ids = new Set();
  filteredNodes().forEach((node) => addIds(ids, node.source_refs));

  state.data.pipelines[0].stages.forEach((stage) => {
    if (state.filters.implementedOnly && stage.status === "planned") {
      return;
    }
    if (!state.filters.showRoadmap && stage.status === "planned") {
      return;
    }
    if (!state.filters.showLlm && stage.llm_config_key) {
      return;
    }
    addIds(ids, stage.source_refs);
  });

  state.data.data_models.forEach((model) => addIds(ids, model.source_refs));
  if (state.filters.showStorage) {
    state.data.storage_map.forEach((entry) => addIds(ids, entry.source_refs));
  }
  state.data.architect_notes.forEach((note) => addIds(ids, note.source_refs));
  return [...ids];
}

function addIds(set, ids) {
  (ids || []).forEach((id) => set.add(id));
}

function createCardButton({ key, title, summary, status, meta, tags = [], className }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `${className} status-${status}`;
  button.dataset.selectionKey = key;

  const top = document.createElement("div");
  top.className = "card-topline";
  top.innerHTML = `
    <h3 class="card-title">${escapeHtml(title)}</h3>
    <span class="status-badge" data-status="${escapeHtml(status)}">${escapeHtml(status)}</span>
  `;

  const summaryEl = document.createElement("p");
  summaryEl.className = "card-summary";
  summaryEl.textContent = summary;

  const metaWrap = document.createElement("div");
  metaWrap.className = "card-meta";
  meta.forEach((item) => {
    const span = document.createElement("span");
    span.className = "badge";
    span.textContent = item;
    metaWrap.appendChild(span);
  });

  const tagsWrap = document.createElement("div");
  tagsWrap.className = "card-tags";
  tags.slice(0, 4).forEach((tag) => {
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = tag;
    tagsWrap.appendChild(span);
  });

  button.append(top, summaryEl);
  if (meta.length) {
    button.appendChild(metaWrap);
  }
  if (tags.length) {
    button.appendChild(tagsWrap);
  }
  return button;
}

function showNodeDetail(nodeId) {
  const node = state.data.nodes.find((item) => item.id === nodeId);
  const relatedEdges = state.data.edges.filter((edge) => edge.from === nodeId || edge.to === nodeId);
  showDetail({
    key: `node:${node.id}`,
    title: node.label,
    status: node.status,
    summary: node.summary,
    attributes: {
      Kind: node.kind,
      Layer: node.layer,
      Tags: node.tags.join(", "),
    },
    relations: relatedEdges.map((edge) => describeEdge(nodeId, edge)),
    sourceIds: node.source_refs,
  });
}

function showPipelineDetail(stageId) {
  const pipeline = state.data.pipelines[0];
  const stage = pipeline.stages.find((item) => item.id === stageId);
  showDetail({
    key: `pipeline:${stage.id}`,
    title: stage.label,
    status: stage.status,
    summary: stage.summary,
    attributes: {
      Inputs: stage.inputs.join(", ") || "none",
      Outputs: stage.outputs.join(", ") || "none",
      Writes: stage.writes.join(", ") || "none",
      "LLM config": stage.llm_config_key || "none",
    },
    relations: stage.related_nodes || [],
    sourceIds: stage.source_refs,
  });
}

function showModelDetail(modelId) {
  const model = state.data.data_models.find((item) => item.id === modelId);
  showDetail({
    key: `model:${model.id}`,
    title: model.label,
    status: model.status,
    summary: model.summary,
    attributes: Object.fromEntries(model.fields.map((field) => [field.name, field.type])),
    relations: model.relationships,
    sourceIds: model.source_refs,
  });
}

function showStorageDetail(storageId) {
  const entry = state.data.storage_map.find((item) => item.id === storageId);
  showDetail({
    key: `storage:${entry.id}`,
    title: entry.label,
    status: entry.status,
    summary: entry.summary,
    attributes: {
      Path: entry.path,
      Kind: entry.kind,
      Reads: entry.reads_by.join(", ") || "none",
      Writes: entry.writes_by.join(", ") || "none",
    },
    relations: entry.relationships,
    sourceIds: entry.source_refs,
  });
}

function showSourceDetail(sourceId) {
  const ref = state.data.source_refs[sourceId];
  showDetail({
    key: `source:${ref.id}`,
    title: ref.label,
    status: ref.status || "implemented",
    summary: ref.summary,
    attributes: {
      Path: ref.path,
      Type: ref.type,
    },
    relations: ref.covers,
    sourceIds: [ref.id],
  });
}

function showNoteDetail(noteId) {
  const note = state.data.architect_notes.find((item) => item.id === noteId);
  showDetail({
    key: `note:${note.id}`,
    title: note.label,
    status: note.status,
    summary: note.summary,
    attributes: {
      Focus: note.tags.join(", "),
      Scope: note.scope,
    },
    relations: note.points,
    sourceIds: note.source_refs,
  });
}

function showDetail({ key, title, status, summary, attributes, relations, sourceIds }) {
  state.selectedKey = key;
  dom.detailTitle.textContent = title;
  dom.detailStatus.textContent = status;
  dom.detailStatus.dataset.status = status;
  dom.detailSummary.textContent = summary;
  dom.detailAttributes.innerHTML = "";
  dom.detailRelations.innerHTML = "";
  dom.detailSources.innerHTML = "";

  Object.entries(attributes).forEach(([term, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dom.detailAttributes.append(dt, dd);
  });

  (relations || []).forEach((relation) => {
    const li = document.createElement("li");
    li.textContent = relation;
    dom.detailRelations.appendChild(li);
  });

  (sourceIds || []).forEach((sourceId) => {
    const ref = state.data.source_refs[sourceId];
    if (!ref) {
      return;
    }
    const li = document.createElement("li");
    li.textContent = `${ref.label} -> ${ref.path}`;
    dom.detailSources.appendChild(li);
  });

  syncSelectionStyles();
}

function seedDetailPanel() {
  const firstNode = state.data.nodes.find((node) => node.id === "engine_pipeline");
  if (firstNode) {
    showNodeDetail(firstNode.id);
  }
}

function syncSelectionStyles() {
  document.querySelectorAll("[data-selection-key]").forEach((element) => {
    element.classList.toggle("is-selected", element.dataset.selectionKey === state.selectedKey);
  });
}

function describeEdge(nodeId, edge) {
  const otherId = edge.from === nodeId ? edge.to : edge.from;
  const other = state.data.nodes.find((node) => node.id === otherId);
  const direction = edge.from === nodeId ? edge.kind : `receives ${edge.kind}`;
  return `${direction} ${other ? other.label : otherId}${edge.label ? ` (${edge.label})` : ""}`;
}

function emptyState(message = "No items match the active filters.") {
  const template = document.getElementById("empty-template");
  const fragment = template.content.firstElementChild.cloneNode(true);
  fragment.querySelector("p").textContent = message;
  return fragment;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function debounce(fn, wait) {
  let timeoutId = null;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), wait);
  };
}

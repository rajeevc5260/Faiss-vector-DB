const UI = (() => {
  const BASE_KEY = "vectorDbBaseUrl";
  const state = {
    docsAfterId: null,
    docsHistory: [],
    docsNextAfterId: null,
  };

  const els = {
    baseUrl: document.querySelector("[data-base-url]") || null,
    baseSave: document.querySelector("[data-base-save]") || null,
    toast: document.querySelector("[data-toast]") || null,
    corsNote: document.querySelector("[data-cors-note]") || null,
  };

  function setToast(message, isError = false) {
    if (!els.toast) return;
    els.toast.textContent = message;
    els.toast.classList.remove("hidden");
    els.toast.style.background = isError ? "#7a1111" : "#0b0b0b";
    setTimeout(() => {
      els.toast.classList.add("hidden");
    }, 2600);
  }

  function setLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.classList.add("btn-loading");
      btn.disabled = true;
      if (!btn.querySelector(".spinner")) {
        const spinner = document.createElement("span");
        spinner.className = "spinner";
        btn.appendChild(spinner);
      }
    } else {
      btn.classList.remove("btn-loading");
      btn.disabled = false;
      const spinner = btn.querySelector(".spinner");
      if (spinner) spinner.remove();
    }
  }

  async function runWithLoading(btn, fn) {
    setLoading(btn, true);
    try {
      return await fn();
    } finally {
      setLoading(btn, false);
    }
  }

  function getBaseUrl() {
    return (els.baseUrl?.value || "").trim().replace(/\/$/, "");
  }

  function updateCorsNote() {
    if (!els.corsNote) return;
    try {
      const base = getBaseUrl();
      if (!base) {
        els.corsNote.classList.add("hidden");
        return;
      }
      const baseOrigin = new URL(base).origin;
      if (baseOrigin !== window.location.origin) {
        els.corsNote.classList.remove("hidden");
      } else {
        els.corsNote.classList.add("hidden");
      }
    } catch {
      els.corsNote.classList.remove("hidden");
    }
  }

  async function apiFetch(path, options = {}) {
    const base = getBaseUrl();
    if (!base) {
      throw new Error("Set API base URL first");
    }
    const url = `${base}${path}`;
    const response = await fetch(url, {
      mode: "cors",
      ...options,
    });
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { raw: text };
      }
    }
    if (!response.ok) {
      throw new Error(data ? JSON.stringify(data) : response.statusText);
    }
    return data;
  }

  function safeJsonParse(raw) {
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return { __invalid: true };
    }
  }

  function buildMetadata(tagsRaw, sourceRaw, advancedRaw) {
    const metadata = {};
    const tags = (tagsRaw || "")
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    if (tags.length) metadata.tags = tags;
    if (sourceRaw) metadata.source = sourceRaw.trim();
    const advanced = safeJsonParse(advancedRaw);
    if (advanced && advanced.__invalid) {
      throw new Error("Advanced metadata JSON is invalid");
    }
    if (advanced) Object.assign(metadata, advanced);
    return metadata;
  }

  function formatDate(value) {
    if (!value) return "—";
    const d = new Date(Number(value));
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  }

  function mountBaseUrl() {
    const stored = localStorage.getItem(BASE_KEY);
    if (els.baseUrl) {
      els.baseUrl.value = stored || "http://localhost:5001";
    }
    if (els.baseSave) {
      els.baseSave.addEventListener("click", () => {
        localStorage.setItem(BASE_KEY, getBaseUrl());
        updateCorsNote();
        setToast("Base URL saved");
      });
    }
    updateCorsNote();
  }

  function bindModalClosers() {
    document.querySelectorAll("[data-modal-close]").forEach((btn) => {
      btn.addEventListener("click", () => closeModal(btn.dataset.modalClose));
    });
  }

  function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
      modal.classList.remove("hidden");
    }
  }

  function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
      modal.classList.add("hidden");
    }
  }

  return {
    apiFetch,
    setToast,
    mountBaseUrl,
    buildMetadata,
    formatDate,
    state,
    openModal,
    closeModal,
    bindModalClosers,
    setLoading,
    runWithLoading,
  };
})();

const NamespaceHome = (() => {
  const listEl = document.querySelector("[data-ns-list]");
  const emptyEl = document.querySelector("[data-ns-empty]");
  const refreshBtn = document.querySelector("[data-ns-refresh]");
  const createBtn = document.querySelector("[data-ns-create]");
  const createInput = document.querySelector("[data-ns-input]");
  const createSubmit = document.querySelector("[data-ns-submit]");

  function renderNamespaces(namespaces) {
    if (!listEl) return;
    listEl.innerHTML = "";
    if (!namespaces.length) {
      if (emptyEl) emptyEl.classList.remove("hidden");
      return;
    }
    if (emptyEl) emptyEl.classList.add("hidden");
    namespaces.forEach((name) => {
      const row = document.createElement("div");
      row.className = "grid grid-cols-[1fr_auto] items-center border-b border-black/10 py-3";

      const title = document.createElement("div");
      title.className = "text-sm font-semibold";
      title.textContent = name;

      const action = document.createElement("a");
      action.className = "btn-outline px-4 py-2 text-xs uppercase tracking-wide";
      action.textContent = "Open";
      action.href = `namespace.html?name=${encodeURIComponent(name)}`;

      row.appendChild(title);
      row.appendChild(action);
      listEl.appendChild(row);
    });
  }

  async function loadNamespaces(btn) {
    try {
      await UI.runWithLoading(btn, async () => {
        UI.setToast("Loading namespaces...");
        const data = await UI.apiFetch("/admin/namespaces");
        renderNamespaces(data.namespaces || []);
        UI.setToast("Namespaces loaded");
      });
    } catch (err) {
      UI.setToast(`Error: ${err.message}`, true);
    }
  }

  async function createNamespace(btn) {
    if (!createInput) return;
    const namespace = createInput.value.trim();
    if (!namespace) {
      UI.setToast("Enter a namespace name", true);
      return;
    }
    try {
      await UI.runWithLoading(btn, async () => {
        await UI.apiFetch("/admin/namespace/create", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ namespace }),
        });
        UI.setToast("Namespace created");
        UI.closeModal("modalCreateNs");
        createInput.value = "";
        loadNamespaces();
      });
    } catch (err) {
      UI.setToast(`Error: ${err.message}`, true);
    }
  }

  function init() {
    if (!listEl) return;
    if (refreshBtn) refreshBtn.addEventListener("click", () => loadNamespaces(refreshBtn));
    if (createBtn) createBtn.addEventListener("click", () => UI.openModal("modalCreateNs"));
    if (createSubmit) createSubmit.addEventListener("click", () => createNamespace(createSubmit));
    loadNamespaces();
  }

  return { init };
})();

const NamespaceDetail = (() => {
  const container = document.querySelector("[data-namespace-detail]");
  if (!container) return { init: () => {} };

  const nsName = new URLSearchParams(window.location.search).get("name");
  const nameEl = document.querySelector("[data-namespace-name]");
  const nameBadge = document.querySelector("[data-namespace-badge]");
  const statsEl = {
    docs: document.querySelector("[data-stat-docs]"),
    inIndex: document.querySelector("[data-stat-in-index]"),
    staging: document.querySelector("[data-stat-staging]"),
    pool: document.querySelector("[data-stat-pool]"),
    trained: document.querySelector("[data-stat-trained]"),
    index: document.querySelector("[data-stat-index]"),
  };

  const docsTable = document.querySelector("[data-docs-table]");
  const docsShard = document.querySelector("[data-docs-shard]");
  const docsLimit = document.querySelector("[data-docs-limit]");
  const docsNext = document.querySelector("[data-docs-next]");
  const docsPrev = document.querySelector("[data-docs-prev]");
  const docsCursor = document.querySelector("[data-docs-cursor]");

  function setNamespaceHeader() {
    if (nameEl) nameEl.textContent = nsName || "Pick a namespace";
    if (nameBadge) nameBadge.textContent = nsName || "—";
  }

  async function loadStats(btn) {
    if (!nsName) return;
    try {
      await UI.runWithLoading(btn, async () => {
        const params = new URLSearchParams({ namespace: nsName });
        const data = await UI.apiFetch(`/admin/namespace?${params.toString()}`);
        const counts = data.totals || data.counts || {};
        statsEl.docs.textContent = counts.docs_total ?? "0";
        statsEl.inIndex.textContent = counts.docs_in_index ?? "0";
        statsEl.staging.textContent = counts.staging_vectors ?? "0";
        statsEl.pool.textContent = counts.training_pool ?? "0";
        const index = data.index || (data.shards && data.shards[0]?.response?.index) || {};
        statsEl.trained.textContent = index.trained_template ? "Trained" : "Not trained";
        statsEl.index.textContent = index.index_exists ? "Index ready" : "No index";
      });
    } catch (err) {
      UI.setToast(`Error: ${err.message}`, true);
    }
  }

  function renderDocs(rows) {
    if (!docsTable) return;
    docsTable.innerHTML = "";
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.className = "px-3 py-4 text-sm text-black/50";
      cell.textContent = "No Records found.";
      row.appendChild(cell);
      docsTable.appendChild(row);
      return;
    }
    rows.forEach((doc) => {
      const row = document.createElement("tr");
      row.className = "border-b border-black/10 text-sm";
      const cells = [
        doc.id || "—",
        doc.type || "—",
        doc.in_index,
        doc.faiss_id ?? "—",
        UI.formatDate(doc.updated_at),
      ];
      cells.forEach((value) => {
        const cell = document.createElement("td");
        cell.className = "px-3 py-2";
        cell.textContent = String(value);
        row.appendChild(cell);
      });
      const actionCell = document.createElement("td");
      actionCell.className = "px-3 py-2";
      const btn = document.createElement("button");
      btn.className = "btn-outline px-3 py-1 text-[11px] uppercase tracking-wide";
      btn.textContent = "Delete";
      btn.dataset.deleteId = doc.id || "";
      actionCell.appendChild(btn);
      row.appendChild(actionCell);
      docsTable.appendChild(row);
    });
  }

  async function loadDocs(btn) {
    if (!nsName) return;
    await UI.runWithLoading(btn, async () => {
      const shard = docsShard?.value || "0";
      const limit = docsLimit?.value || "50";
      const params = new URLSearchParams({ namespace: nsName, shard, limit });
      if (UI.state.docsAfterId) params.set("after_id", UI.state.docsAfterId);
      const data = await UI.apiFetch(`/admin/namespace/docs?${params.toString()}`);
      const payload = data?.response?.response || data;
      renderDocs(payload.docs || []);
      UI.state.docsNextAfterId = payload.next_after_id || null;
      if (docsCursor) docsCursor.textContent = UI.state.docsAfterId || "start";
    });
  }

  function nextDocs(btn) {
    if (!UI.state.docsNextAfterId) return;
    UI.state.docsHistory.push(UI.state.docsAfterId);
    UI.state.docsAfterId = UI.state.docsNextAfterId;
    loadDocs(btn);
  }

  function prevDocs(btn) {
    if (!UI.state.docsHistory.length) return;
    UI.state.docsAfterId = UI.state.docsHistory.pop() || null;
    loadDocs(btn);
  }

  async function insertText(btn) {
    const textInput = document.querySelector("[data-insert-text]");
    const tagsInput = document.querySelector("[data-insert-tags]");
    const sourceInput = document.querySelector("[data-insert-source]");
    const advancedInput = document.querySelector("[data-insert-meta]");
    const idInput = document.querySelector("[data-insert-id]");
    if (!textInput) return;
    let id = idInput?.value.trim();
    if (!id) {
      id = `doc_${Math.random().toString(16).slice(2, 10)}`;
      if (idInput) idInput.value = id;
    }
    try {
      const metadata = UI.buildMetadata(tagsInput?.value, sourceInput?.value, advancedInput?.value);
      const payload = {
        namespace: nsName,
        id,
        text: textInput.value.trim(),
        metadata,
      };
      return await UI.runWithLoading(btn, async () => {
        const data = await UI.apiFetch("/insert", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        UI.setToast(`Inserted ${id}`);
        textInput.value = "";
        UI.closeModal("modalInsertText");
        loadStats();
        loadDocs();
        return data;
      });
    } catch (err) {
      UI.setToast(`Error: ${err.message}`, true);
    }
  }

  async function insertImage(btn) {
    const idInput = document.querySelector("[data-insert-image-id]");
    const fileInput = document.querySelector("[data-insert-image-file]");
    const tagsInput = document.querySelector("[data-insert-image-tags]");
    const sourceInput = document.querySelector("[data-insert-image-source]");
    const advancedInput = document.querySelector("[data-insert-image-meta]");
    if (!fileInput || !fileInput.files.length) {
      UI.setToast("Choose an image file", true);
      return;
    }
    let id = idInput?.value.trim();
    if (!id) {
      id = `img_${Math.random().toString(16).slice(2, 10)}`;
      if (idInput) idInput.value = id;
    }
    const metadata = UI.buildMetadata(tagsInput?.value, sourceInput?.value, advancedInput?.value);
    const form = new FormData();
    form.append("namespace", nsName);
    form.append("id", id);
    if (Object.keys(metadata).length) form.append("metadata", JSON.stringify(metadata));
    form.append("image", fileInput.files[0]);
    try {
      await UI.runWithLoading(btn, async () => {
        await UI.apiFetch("/insert", { method: "POST", body: form });
        UI.setToast(`Inserted ${id}`);
        fileInput.value = "";
        UI.closeModal("modalInsertImage");
        loadStats();
        loadDocs();
      });
    } catch (err) {
      UI.setToast(`Error: ${err.message}`, true);
    }
  }

  async function searchText(btn) {
    const queryInput = document.querySelector("[data-search-query]");
    const topKInput = document.querySelector("[data-search-topk]");
    const resultsWrap = document.querySelector("[data-search-results]");
    if (!queryInput || !resultsWrap) return;
    const payload = {
      namespace: nsName,
      query: queryInput.value.trim(),
    };
    if (topKInput?.value) payload.top_k = parseInt(topKInput.value, 10);
    await UI.runWithLoading(btn, async () => {
      const data = await UI.apiFetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderSearchResults(resultsWrap, data.results || []);
    });
  }

  async function searchImage(btn) {
    const fileInput = document.querySelector("[data-search-image-file]");
    const topKInput = document.querySelector("[data-search-image-topk]");
    const resultsWrap = document.querySelector("[data-search-results]");
    if (!fileInput || !fileInput.files.length || !resultsWrap) {
      UI.setToast("Choose an image to search", true);
      return;
    }
    const form = new FormData();
    form.append("namespace", nsName);
    if (topKInput?.value) form.append("top_k", topKInput.value);
    form.append("image", fileInput.files[0]);
    await UI.runWithLoading(btn, async () => {
      const data = await UI.apiFetch("/search/image", { method: "POST", body: form });
      renderSearchResults(resultsWrap, data.results || []);
    });
  }

  function renderSearchResults(container, results) {
    container.innerHTML = "";
    if (!results.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.className = "px-3 py-3 text-sm text-black/60";
      cell.textContent = "No results yet.";
      row.appendChild(cell);
      container.appendChild(row);
      return;
    }
    results.forEach((item) => {
      const row = document.createElement("tr");
      row.className = "border-b border-black/10 text-xs";
      const snippet = item.text ? String(item.text).slice(0, 280) : "No preview";
      const cells = [
        { value: item.id || "unknown" },
        { value: item.type || "text" },
        { value: Number(item.score || 0).toFixed(4) },
      ];
      cells.forEach((entry) => {
        const cell = document.createElement("td");
        cell.className = "px-3 py-2";
        cell.textContent = String(entry.value);
        row.appendChild(cell);
      });
      const snippetCell = document.createElement("td");
      snippetCell.className = "px-3 py-2";
      const snippetEl = document.createElement("div");
      snippetEl.className = "line-clamp-2 max-w-md";
      snippetEl.textContent = snippet;
      snippetEl.title = snippet;
      snippetCell.appendChild(snippetEl);
      row.appendChild(snippetCell);
      container.appendChild(row);
    });
  }

  async function deleteDoc(id, btn) {
    if (!id) return;
    await UI.runWithLoading(btn, async () => {
      await UI.apiFetch("/delete", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ namespace: nsName, id }),
      });
      UI.setToast(`Deleted ${id}`);
      loadStats();
      loadDocs();
    });
  }

  async function retrain(btn) {
    const lastInput = document.querySelector("[data-retrain-last]");
    const payload = { namespace: nsName };
    if (lastInput?.value) payload.last_n = parseInt(lastInput.value, 10);
    await UI.runWithLoading(btn, async () => {
      await UI.apiFetch("/admin/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      UI.setToast("Retrain requested");
      UI.closeModal("modalRetrain");
      loadStats();
    });
  }

  async function deleteNamespace(btn) {
    await UI.runWithLoading(btn, async () => {
      await UI.apiFetch("/admin/namespace", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ namespace: nsName }),
      });
      UI.setToast("Namespace deleted");
      window.location.href = "index.html";
    });
  }

  function init() {
    setNamespaceHeader();
    loadStats();
    loadDocs();
    if (docsNext) docsNext.addEventListener("click", () => nextDocs(docsNext));
    if (docsPrev) docsPrev.addEventListener("click", () => prevDocs(docsPrev));

    const refreshBtn = document.querySelector("[data-docs-refresh]");
    refreshBtn?.addEventListener("click", () => loadDocs(refreshBtn));
    const insertTextBtn = document.querySelector("[data-insert-text-submit]");
    insertTextBtn?.addEventListener("click", () => insertText(insertTextBtn));
    const insertImageBtn = document.querySelector("[data-insert-image-submit]");
    insertImageBtn?.addEventListener("click", () => insertImage(insertImageBtn));
    const searchBtn = document.querySelector("[data-search-submit]");
    searchBtn?.addEventListener("click", () => searchText(searchBtn));
    const searchImageBtn = document.querySelector("[data-search-image-submit]");
    searchImageBtn?.addEventListener("click", () => searchImage(searchImageBtn));
    docsTable?.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const id = target.dataset.deleteId;
      if (id) {
        deleteDoc(id, target);
      }
    });

    document.querySelector("[data-open-retrain]")?.addEventListener("click", () => UI.openModal("modalRetrain"));
    const retrainBtn = document.querySelector("[data-retrain-submit]");
    retrainBtn?.addEventListener("click", () => retrain(retrainBtn));
    document.querySelector("[data-open-delete-ns]")?.addEventListener("click", () => UI.openModal("modalDeleteNs"));
    const deleteNsBtn = document.querySelector("[data-delete-ns-confirm]");
    deleteNsBtn?.addEventListener("click", () => deleteNamespace(deleteNsBtn));
    document.querySelector("[data-open-insert-text]")?.addEventListener("click", () => UI.openModal("modalInsertText"));
    document.querySelector("[data-open-insert-image]")?.addEventListener("click", () => UI.openModal("modalInsertImage"));

    const toggles = document.querySelectorAll("[data-search-toggle]");
    const panelText = document.querySelector("[data-search-panel=\"text\"]");
    const panelImage = document.querySelector("[data-search-panel=\"image\"]");
    const setMode = (mode) => {
      if (panelText && panelImage) {
        panelText.classList.toggle("hidden", mode !== "text");
        panelImage.classList.toggle("hidden", mode !== "image");
      }
      toggles.forEach((btn) => {
        const active = btn.dataset.searchToggle === mode;
        btn.classList.toggle("btn-primary", active);
        btn.classList.toggle("btn-outline", !active);
      });
    };
    toggles.forEach((btn) => {
      btn.addEventListener("click", () => setMode(btn.dataset.searchToggle));
    });
    setMode("text");
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", () => {
  UI.mountBaseUrl();
  UI.bindModalClosers();
  NamespaceHome.init();
  NamespaceDetail.init();
});

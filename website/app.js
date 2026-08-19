(() => {
  "use strict";

  const SESSION_KEY = "dora_session_id";

  const messagesEl = document.getElementById("messages");
  const composerEl = document.getElementById("composer");
  const inputEl = document.getElementById("composer-input");
  const sendBtn = document.getElementById("composer-send");
  const healthDot = document.getElementById("health-dot");
  const newSessionBtn = document.getElementById("new-session");
  const quickPrompts = document.getElementById("quick-prompts");

  const uploadForm = document.getElementById("upload-form");
  const uploadPackageIdInput = document.getElementById("upload-package-id");
  const uploadFilesInput = document.getElementById("upload-files");
  const uploadDrop = document.getElementById("upload-drop");
  const uploadDropText = document.getElementById("upload-drop-text");
  const uploadSubmitBtn = document.getElementById("upload-submit");
  const uploadStatusEl = document.getElementById("upload-status");

  let sessionId = sessionStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, sessionId);
  }

  let busy = false;
  let lastPackageId = null; // most recently uploaded package, used as the quick-prompt fill-in

  // ---- markdown-ish rendering (escape first, then a few safe conversions) ----

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function inline(s) {
    return escapeHtml(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  }

  function isTableBlock(lines) {
    if (lines.length < 2) return false;
    if (!lines[0].includes("|")) return false;
    return /^\s*\|?[\s:-]+\|[\s:|-]+$/.test(lines[1]);
  }

  function renderTable(lines) {
    const split = (row) =>
      row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    const header = split(lines[0]);
    const bodyRows = lines.slice(2).map(split);
    let html = "<table><thead><tr>";
    header.forEach((h) => (html += `<th>${inline(h)}</th>`));
    html += "</tr></thead><tbody>";
    bodyRows.forEach((row) => {
      html += "<tr>";
      row.forEach((c) => (html += `<td>${inline(c)}</td>`));
      html += "</tr>";
    });
    html += "</tbody></table>";
    return html;
  }

  function renderMarkdownish(text) {
    const blocks = text.split(/\n{2,}/);
    return blocks
      .map((block) => {
        const lines = block.split("\n").filter((l) => l.length > 0);
        if (isTableBlock(lines)) return renderTable(lines);
        return `<p>${inline(block).replace(/\n/g, "<br>")}</p>`;
      })
      .join("");
  }

  // ---- message rendering ----

  function addMessage(role, text, { pending = false, error = false } = {}) {
    const wrap = document.createElement("div");
    wrap.className = `msg msg-${role}${pending ? " msg-pending" : ""}${error ? " msg-error" : ""}`;

    const roleEl = document.createElement("div");
    roleEl.className = "msg-role";
    roleEl.textContent = role === "user" ? "You" : "DORA";
    wrap.appendChild(roleEl);

    const bodyEl = document.createElement("div");
    bodyEl.className = "msg-body";
    if (pending) {
      bodyEl.textContent = text;
    } else if (role === "assistant" && !error) {
      bodyEl.innerHTML = renderMarkdownish(text);
    } else {
      bodyEl.textContent = text;
    }
    wrap.appendChild(bodyEl);

    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  function addTrace(wrap, trace) {
    if (!trace || trace.length === 0) return;
    const details = document.createElement("details");
    details.className = "trace";
    const summary = document.createElement("summary");
    summary.textContent = `${trace.length} tool call${trace.length === 1 ? "" : "s"} — audit trail`;
    details.appendChild(summary);

    const list = document.createElement("div");
    list.className = "trace-list";
    trace.forEach((t) => {
      const item = document.createElement("div");
      item.className = "trace-item";
      const toolEl = document.createElement("div");
      toolEl.className = "trace-tool";
      toolEl.textContent = t.tool;
      const io = document.createElement("pre");
      io.className = "trace-io";
      io.textContent = `input:  ${JSON.stringify(t.input)}\noutput: ${JSON.stringify(t.output, null, 0)}`;
      item.appendChild(toolEl);
      item.appendChild(io);
      list.appendChild(item);
    });
    details.appendChild(list);
    wrap.appendChild(details);
  }

  // ---- API calls ----

  async function loadHealth() {
    try {
      const resp = await fetch("/api/health");
      const data = await resp.json();
      healthDot.classList.remove("ok", "bad");
      healthDot.classList.add(data.model_key_configured ? "ok" : "bad");
      const missingCredHint =
        data.backend === "ces"
          ? "No Google Cloud credentials found on the server (run `gcloud auth application-default login`)"
          : "GOOGLE_API_KEY is not set on the server yet";
      healthDot.title = data.model_key_configured
        ? `Connected — ${data.backend} backend, model ${data.model}`
        : missingCredHint;
    } catch {
      healthDot.classList.add("bad");
      healthDot.title = "Server unreachable";
    }
  }


  // ---- package upload ----

  function setUploadStatus(text, kind) {
    uploadStatusEl.hidden = !text;
    uploadStatusEl.textContent = text;
    uploadStatusEl.className = `upload-status${kind ? ` ${kind}` : ""}`;
  }

  function describeSelectedFiles(fileList) {
    if (!fileList || fileList.length === 0) {
      uploadDropText.textContent = "Choose package folder…";
      uploadSubmitBtn.disabled = true;
      return;
    }

    const files = Array.from(fileList);
    const pdfCount = files.filter((f) => f.name.toLowerCase().endsWith(".pdf")).length;
    const hasMetadata = files.some((f) => f.name === "Project_Metadata.json");
    const folderName = files[0].webkitRelativePath
      ? files[0].webkitRelativePath.split("/")[0]
      : null;

    const label = folderName || (files.length === 1 ? files[0].name : `${files.length} files`);
    uploadDropText.textContent = hasMetadata
      ? `${label} — ${pdfCount} PDF${pdfCount === 1 ? "" : "s"}`
      : `${label} — ⚠ no Project_Metadata.json found`;

    // Convenience: default the package id to the folder name, without stomping
    // anything the user already typed.
    if (folderName && !uploadPackageIdInput.value.trim()) {
      uploadPackageIdInput.value = folderName.replace(/[^A-Za-z0-9_-]/g, "-").slice(0, 64);
    }

    uploadSubmitBtn.disabled = false;
  }

  uploadFilesInput.addEventListener("change", () => {
    describeSelectedFiles(uploadFilesInput.files);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    uploadDrop.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadDrop.classList.add("dragover");
    })
  );
  ["dragleave", "dragend"].forEach((evt) =>
    uploadDrop.addEventListener(evt, () => uploadDrop.classList.remove("dragover"))
  );
  uploadDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDrop.classList.remove("dragover");
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFilesInput.files = e.dataTransfer.files;
      describeSelectedFiles(uploadFilesInput.files);
    }
  });

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileList = uploadFilesInput.files;
    if (!fileList || fileList.length === 0) return;

    uploadSubmitBtn.disabled = true;
    setUploadStatus("Uploading and parsing…", "busy");

    const form = new FormData();
    form.append("package_id", uploadPackageIdInput.value.trim());
    Array.from(fileList).forEach((f) => form.append("files", f));

    try {
      const resp = await fetch("/api/packages/upload", { method: "POST", body: form });
      const data = await resp.json();

      if (!resp.ok) {
        setUploadStatus(data.detail || "Upload failed.", "bad");
        uploadSubmitBtn.disabled = false;
        return;
      }

      setUploadStatus(
        `Loaded ${data.package_id} — ${data.documents.length} document(s), ${data.clauses_extracted} clause(s) extracted.`,
        "ok"
      );
      uploadForm.reset();
      describeSelectedFiles(null);
      lastPackageId = data.package_id;
      addMessage(
        "assistant",
        `Uploaded and parsed **${data.package_id}** (${data.project_title}): ${data.documents.length} document(s), ${data.clauses_extracted} clause(s) resolved to checklist headings. Ask me to review **${data.package_id}** whenever you're ready.`
      );
    } catch (err) {
      setUploadStatus(`Network error: ${err}`, "bad");
      uploadSubmitBtn.disabled = false;
    }
  });

  async function sendMessage(text) {
    busy = true;
    sendBtn.disabled = true;

    addMessage("user", text);
    const pendingWrap = addMessage("assistant", "Reviewing…", { pending: true });

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data = await resp.json();

      pendingWrap.remove();

      if (!resp.ok) {
        addMessage("assistant", data.detail || "Request failed.", { error: true });
        return;
      }

      const wrap = addMessage("assistant", data.reply || "(empty response)");
      addTrace(wrap, data.trace);
    } catch (err) {
      pendingWrap.remove();
      addMessage("assistant", `Network error: ${err}`, { error: true });
    } finally {
      busy = false;
      sendBtn.disabled = false;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  // ---- events ----

  composerEl.addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) return;
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    inputEl.style.height = "auto";
    sendMessage(text);
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerEl.requestSubmit();
    }
  });

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
  });

  quickPrompts.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-template]");
    if (!btn) return;
    const pkg = lastPackageId || "<PACKAGE_ID>";
    inputEl.value = btn.dataset.template.replace("{package}", pkg);
    inputEl.dispatchEvent(new Event("input"));
    inputEl.focus();
  });

  newSessionBtn.addEventListener("click", async () => {
    if (busy) return;
    try {
      await fetch("/api/session/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch {
      /* best effort */
    }
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, sessionId);
    messagesEl.innerHTML = "";
    addMessage(
      "assistant",
      "New session started. Upload a package or name one in your message, then ask me to review it."
    );
  });

  loadHealth();
})();

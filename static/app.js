document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const docNameText = document.getElementById("docNameText");
    const chunksCountText = document.getElementById("chunksCountText");
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadProgress = document.getElementById("uploadProgress");
    const uploadStatusText = document.getElementById("uploadStatusText");
    const chatFeed = document.getElementById("chatFeed");
    const queryForm = document.getElementById("queryForm");
    const questionInput = document.getElementById("questionInput");
    const sendBtn = document.getElementById("sendBtn");
    const toastContainer = document.getElementById("toastContainer");

    // Configure Marked.js
    if (window.marked) {
        marked.setOptions({
            breaks: true,
            gfm: true
        });
    }

    // 1. Fetch initial status
    fetchStatus();

    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                docNameText.textContent = data.active_document || "No document";
                chunksCountText.textContent = `${data.total_chunks || 0} Chunks`;
            }
        } catch (err) {
            console.warn("Could not fetch status:", err);
            docNameText.textContent = "Offline / Idle";
        }
    }

    // 2. Drag and Drop File Upload Event Listeners
    uploadZone.addEventListener("click", () => fileInput.click());

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'txt'].includes(ext)) {
            showToast("Only PDF and TXT files are supported.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("chunk_strategy", "Recursive Character");
        formData.append("chunk_size", 1000);
        formData.append("chunk_overlap", 200);

        uploadZone.style.display = "none";
        uploadProgress.style.display = "flex";
        uploadStatusText.textContent = `Ingesting ${file.name}...`;

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Upload failed");
            }

            const data = await res.json();
            docNameText.textContent = data.document_name;
            chunksCountText.textContent = `${data.total_chunks} Chunks`;

            showToast(`Successfully ingested ${data.document_name}!`, "success");
        } catch (err) {
            console.error("Upload error:", err);
            showToast(`Upload Error: ${err.message}`, "error");
        } finally {
            uploadProgress.style.display = "none";
            uploadZone.style.display = "block";
            fileInput.value = "";
        }
    }

    // 3. Prompt Fill Helper
    window.fillPrompt = function (text) {
        questionInput.value = text;
        questionInput.focus();
    };

    // 4. Query Submission Handler
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;

        // Append User Message Card
        appendUserMessage(question);
        questionInput.value = "";
        sendBtn.disabled = true;

        // Create AI Loading Message Card
        const aiCardId = `ai-msg-${Date.now()}`;
        appendAILoadingCard(aiCardId);

        try {
            const res = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    chunk_strategy: "Recursive Character",
                    query_mode: "Multi-Query Expansion",
                    use_hybrid: true,
                    use_reranker: true,
                    model_choice: "gemini-3.6-flash",
                    temperature: 0.0,
                    retrieval_k: 6,
                    rerank_k: 3
                })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Query failed");
            }

            const data = await res.json();
            updateAICardContent(aiCardId, data);
        } catch (err) {
            console.error("Query Error:", err);
            updateAICardError(aiCardId, err.message);
            showToast(`Query Error: ${err.message}`, "error");
        } finally {
            sendBtn.disabled = false;
            scrollToBottom();
        }
    });

    function appendUserMessage(text) {
        const card = document.createElement("div");
        card.className = "message-card user-card glass-card";
        card.innerHTML = `
            <div class="card-header">
                <div class="card-header-left">
                    <span class="bot-avatar">👤</span>
                    <span class="author-name">You</span>
                </div>
            </div>
            <div class="card-body">
                ${escapeHtml(text)}
            </div>
        `;
        chatFeed.appendChild(card);
        scrollToBottom();
    }

    function appendAILoadingCard(id) {
        const card = document.createElement("div");
        card.className = "message-card ai-card glass-card";
        card.id = id;
        card.innerHTML = `
            <div class="card-header">
                <div class="card-header-left">
                    <span class="bot-avatar">🤖</span>
                    <span class="author-name">RAG AI Assistant</span>
                </div>
            </div>
            <div class="card-body">
                <div class="spinner" style="display: inline-block; margin-right: 8px;"></div>
                <span>Synthesizing answer & retrieving context passages...</span>
            </div>
        `;
        chatFeed.appendChild(card);
        scrollToBottom();
    }

    function updateAICardContent(id, data) {
        const card = document.getElementById(id);
        if (!card) return;

        const renderedMarkdown = window.marked ? marked.parse(data.answer) : escapeHtml(data.answer);

        // Build Source passages items
        let sourcesHTML = "";
        if (data.sources && data.sources.length > 0) {
            sourcesHTML = data.sources.map(s => `
                <div class="source-item">
                    <div class="source-meta">
                        <span>Passage Rank #${s.rank} (Page ${s.page})</span>
                        ${s.relevance_score ? `<span>Relevance: ${s.relevance_score}%</span>` : ''}
                    </div>
                    <div class="source-text">${escapeHtml(s.content)}</div>
                </div>
            `).join("");
        } else {
            sourcesHTML = "<p>No source passages retrieved.</p>";
        }

        // Build Pipeline Debug details
        const debug = data.pipeline_debug;
        const expandedQueriesList = debug.expanded_queries
            .map(q => `<li><code>${escapeHtml(q)}</code></li>`).join("");

        card.innerHTML = `
            <div class="card-header">
                <div class="card-header-left">
                    <span class="bot-avatar">🤖</span>
                    <span class="author-name">RAG AI Assistant</span>
                </div>
                <div class="meta-info">
                    <span class="meta-badge">⚡ ${data.elapsed_time_seconds}s</span>
                    <span class="meta-badge">${data.model_used}</span>
                </div>
            </div>
            <div class="card-body">
                ${renderedMarkdown}
            </div>

            <!-- Source References Accordion -->
            <div class="sources-toggle">
                <button class="toggle-btn" onclick="toggleAccordion('${id}-sources')">
                    <span>📚 View Source References (${data.sources ? data.sources.length : 0})</span>
                    <span id="${id}-sources-arrow">▼</span>
                </button>
                <div class="toggle-content" id="${id}-sources">
                    ${sourcesHTML}
                </div>
            </div>

            <!-- Pipeline Debug Details Accordion -->
            <div class="debug-toggle">
                <button class="toggle-btn" onclick="toggleAccordion('${id}-debug')">
                    <span>⚡ Pipeline Debug Details</span>
                    <span id="${id}-debug-arrow">▼</span>
                </button>
                <div class="toggle-content" id="${id}-debug">
                    <p><strong>Query Strategy:</strong> ${escapeHtml(debug.query_mode)}</p>
                    <p style="margin-top: 6px;"><strong>Search Queries Dispatched:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 4px;">
                        ${expandedQueriesList}
                    </ul>
                    <p style="margin-top: 8px;">
                        <strong>Candidate Hits:</strong> ${debug.candidate_count} | 
                        <strong>Top Reranked:</strong> ${debug.reranked_count}
                    </p>
                </div>
            </div>
        `;
    }

    function updateAICardError(id, errMsg) {
        const card = document.getElementById(id);
        if (!card) return;
        card.innerHTML = `
            <div class="card-header">
                <div class="card-header-left">
                    <span class="bot-avatar">🤖</span>
                    <span class="author-name">RAG AI Assistant</span>
                </div>
            </div>
            <div class="card-body" style="color: #ef4444;">
                <strong>⚠️ Error generating answer:</strong> ${escapeHtml(errMsg)}
            </div>
        `;
    }

    window.toggleAccordion = function (contentId) {
        const content = document.getElementById(contentId);
        const arrow = document.getElementById(`${contentId}-arrow`);
        if (!content) return;

        if (content.classList.contains("open")) {
            content.classList.remove("open");
            if (arrow) arrow.textContent = "▼";
        } else {
            content.classList.add("open");
            if (arrow) arrow.textContent = "▲";
        }
    };

    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 4000);
    }

    function scrollToBottom() {
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});

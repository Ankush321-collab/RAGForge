(() => {
  // frontend/app.jsx
  var { useState, useEffect, useRef } = React;
  var API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "http://localhost:8000" : "";
  function App() {
    const [activeTab, setActiveTab] = useState("chat");
    const [documents, setDocuments] = useState([]);
    const [selectedDocId, setSelectedDocId] = useState("all");
    const [isLoadingDocs, setIsLoadingDocs] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgressMsg, setUploadProgressMsg] = useState("");
    const [services, setServices] = useState({
      backend: "checking",
      minio: "checking",
      opensearch: "checking",
      neo4j: "checking"
    });
    const [chatMessages, setChatMessages] = useState([
      {
        role: "assistant",
        route: "SYSTEM_READY",
        content: "Welcome to RAGForge! Upload any PDF document or choose an indexed document to ask questions, verify citations, and inspect hybrid retrieval.",
        citations: [],
        groundingScore: 1,
        isValid: true
      }
    ]);
    const [inputQuery, setInputQuery] = useState("");
    const [isChatLoading, setIsChatLoading] = useState(false);
    const [activeCitation, setActiveCitation] = useState(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchMode, setSearchMode] = useState("hybrid");
    const [rerank, setRerank] = useState(true);
    const [topK, setTopK] = useState(5);
    const [searchResults, setSearchResults] = useState(null);
    const [isSearching, setIsSearching] = useState(false);
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const [isGraphLoading, setIsGraphLoading] = useState(false);
    useEffect(() => {
      fetchDocuments();
      checkAllServices();
    }, []);
    useEffect(() => {
      if (selectedDocId && selectedDocId !== "all") {
        fetchDocumentGraph(selectedDocId);
      }
    }, [selectedDocId]);
    const fetchDocuments = async () => {
      setIsLoadingDocs(true);
      try {
        const res = await fetch(`${API_BASE}/documents`);
        if (res.ok) {
          const data = await res.json();
          setDocuments(data);
          if (data.length > 0 && selectedDocId === "all") {
          }
        }
      } catch (err) {
        console.warn("Could not fetch documents:", err);
      } finally {
        setIsLoadingDocs(false);
      }
    };
    const checkAllServices = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        setServices((prev) => ({ ...prev, backend: res.ok ? "online" : "offline" }));
      } catch {
        setServices((prev) => ({ ...prev, backend: "offline" }));
      }
      try {
        await fetch("http://localhost:9000/minio/health/live", { mode: "no-cors" });
        setServices((prev) => ({ ...prev, minio: "online" }));
      } catch {
        setServices((prev) => ({ ...prev, minio: "offline" }));
      }
      try {
        await fetch("http://localhost:9200", { mode: "no-cors" });
        setServices((prev) => ({ ...prev, opensearch: "online" }));
      } catch {
        setServices((prev) => ({ ...prev, opensearch: "offline" }));
      }
      try {
        await fetch("http://localhost:7474", { mode: "no-cors" });
        setServices((prev) => ({ ...prev, neo4j: "online" }));
      } catch {
        setServices((prev) => ({ ...prev, neo4j: "offline" }));
      }
    };
    const fetchDocumentGraph = async (docId) => {
      setIsGraphLoading(true);
      try {
        const res = await fetch(`${API_BASE}/documents/${docId}/graph`);
        if (res.ok) {
          const data = await res.json();
          setGraphData(data);
        }
      } catch (err) {
        console.warn("Graph fetch error:", err);
      } finally {
        setIsGraphLoading(false);
      }
    };
    const handleFileUpload = async (file) => {
      if (!file || !file.name.endsWith(".pdf")) {
        alert("Please upload a valid PDF document (.pdf)");
        return;
      }
      setIsUploading(true);
      setUploadProgressMsg(`Ingesting & chunking "${file.name}"...`);
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch(`${API_BASE}/documents`, {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(errText || "Upload failed");
        }
        const newDoc = await res.json();
        setUploadProgressMsg(`Successfully indexed ${newDoc.num_pages} pages into ${newDoc.num_chunks} vector chunks!`);
        await fetchDocuments();
        setSelectedDocId(newDoc.document_id);
        setChatMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            route: "INGESTION_COMPLETE",
            content: `Document **"${newDoc.filename}"** is now fully indexed!

\u2022 **Pages:** ${newDoc.num_pages}
\u2022 **Parent-Child Chunks:** ${newDoc.num_chunks}
\u2022 **Vector Database:** OpenSearch / ChromaDB

You can now ask any question regarding this document.`,
            citations: [],
            groundingScore: 1,
            isValid: true
          }
        ]);
        setTimeout(() => {
          setIsUploading(false);
          setUploadProgressMsg("");
          setActiveTab("chat");
        }, 1500);
      } catch (err) {
        alert(`Document upload error: ${err.message}`);
        setIsUploading(false);
        setUploadProgressMsg("");
      }
    };
    const handleChatSubmit = async (e) => {
      e?.preventDefault();
      if (!inputQuery.trim() || isChatLoading) return;
      const userQ = inputQuery.trim();
      setInputQuery("");
      setChatMessages((prev) => [...prev, { role: "user", content: userQ }]);
      setIsChatLoading(true);
      try {
        const payload = {
          question: userQ,
          k: 5,
          document_id: selectedDocId !== "all" ? selectedDocId : null
        };
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setChatMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            route: data.route,
            content: data.answer,
            citations: data.citations || [],
            groundingScore: data.grounding_score,
            isValid: data.is_valid,
            context: data.context
          }
        ]);
      } catch (err) {
        setChatMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            route: "ERROR",
            content: `Chat request failed: ${err.message}. Ensure the FastAPI backend is running at http://localhost:8000.`,
            citations: [],
            groundingScore: 0,
            isValid: false
          }
        ]);
      } finally {
        setIsChatLoading(false);
      }
    };
    const handleSearch = async (e) => {
      e?.preventDefault();
      if (!searchQuery.trim() || isSearching) return;
      setIsSearching(true);
      try {
        const payload = {
          query: searchQuery.trim(),
          mode: searchMode,
          top_k: topK,
          rerank,
          document_id: selectedDocId !== "all" ? selectedDocId : null
        };
        const res = await fetch(`${API_BASE}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setSearchResults(data);
      } catch (err) {
        alert(`Search failed: ${err.message}`);
      } finally {
        setIsSearching(false);
      }
    };
    const activeDoc = documents.find((d) => d.document_id === selectedDocId);
    return /* @__PURE__ */ React.createElement("div", { className: "app-shell" }, /* @__PURE__ */ React.createElement("header", { className: "app-header" }, /* @__PURE__ */ React.createElement("div", { className: "brand-wrapper" }, /* @__PURE__ */ React.createElement("div", { className: "brand-icon" }, /* @__PURE__ */ React.createElement("svg", { width: "22", height: "22", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2.4", strokeLinecap: "round", strokeLinejoin: "round" }, /* @__PURE__ */ React.createElement("polygon", { points: "12 2 2 7 12 12 22 7 12 2" }), /* @__PURE__ */ React.createElement("polyline", { points: "2 17 12 22 22 17" }), /* @__PURE__ */ React.createElement("polyline", { points: "2 12 12 17 22 12" }))), /* @__PURE__ */ React.createElement("div", { className: "brand-text" }, /* @__PURE__ */ React.createElement("h1", null, "RAG", /* @__PURE__ */ React.createElement("span", null, "Forge")), /* @__PURE__ */ React.createElement("p", null, "Multimodal Agentic RAG Platform \u2022 Production Architecture"))), /* @__PURE__ */ React.createElement("div", { className: "infra-strip" }, /* @__PURE__ */ React.createElement("div", { className: "status-pill", title: "FastAPI API Gateway" }, /* @__PURE__ */ React.createElement("span", { className: `status-dot ${services.backend}` }), /* @__PURE__ */ React.createElement("span", null, "API :8000")), /* @__PURE__ */ React.createElement("div", { className: "status-pill", title: "MinIO S3 Object Storage" }, /* @__PURE__ */ React.createElement("span", { className: `status-dot ${services.minio}` }), /* @__PURE__ */ React.createElement("span", null, "MinIO :9000")), /* @__PURE__ */ React.createElement("div", { className: "status-pill", title: "OpenSearch Vector DB" }, /* @__PURE__ */ React.createElement("span", { className: `status-dot ${services.opensearch}` }), /* @__PURE__ */ React.createElement("span", null, "OpenSearch :9200")), /* @__PURE__ */ React.createElement("div", { className: "status-pill", title: "Neo4j Knowledge Graph" }, /* @__PURE__ */ React.createElement("span", { className: `status-dot ${services.neo4j}` }), /* @__PURE__ */ React.createElement("span", null, "Neo4j :7474")), /* @__PURE__ */ React.createElement("button", { className: "btn-icon", onClick: checkAllServices, title: "Refresh Service Status" }, /* @__PURE__ */ React.createElement("svg", { width: "15", height: "15", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("path", { d: "M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" }))))), /* @__PURE__ */ React.createElement("div", { className: "document-context-bar" }, /* @__PURE__ */ React.createElement("div", { className: "doc-selector-group" }, /* @__PURE__ */ React.createElement("span", { className: "doc-selector-label" }, "Corpus Target:"), /* @__PURE__ */ React.createElement(
      "select",
      {
        className: "doc-dropdown",
        value: selectedDocId,
        onChange: (e) => setSelectedDocId(e.target.value)
      },
      /* @__PURE__ */ React.createElement("option", { value: "all" }, "\u{1F310} All Indexed Documents (", documents.length, ")"),
      documents.map((d) => /* @__PURE__ */ React.createElement("option", { key: d.document_id, value: d.document_id }, "\u{1F4C4} ", d.filename, " (", d.num_pages, " pages \u2022 ", d.num_chunks, " chunks)"))
    )), /* @__PURE__ */ React.createElement("div", { className: "doc-stats-summary" }, activeDoc ? /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("span", { className: "doc-stat-item" }, "Pages: ", /* @__PURE__ */ React.createElement("strong", null, activeDoc.num_pages)), /* @__PURE__ */ React.createElement("span", { className: "doc-stat-item" }, "Vector Chunks: ", /* @__PURE__ */ React.createElement("strong", null, activeDoc.num_chunks)), /* @__PURE__ */ React.createElement("span", { className: "doc-stat-item" }, "Size: ", /* @__PURE__ */ React.createElement("strong", null, (activeDoc.size_bytes / 1024).toFixed(1), " KB")), /* @__PURE__ */ React.createElement("span", { className: "doc-stat-item" }, "Status: ", /* @__PURE__ */ React.createElement("strong", { style: { color: "#059669" } }, "Indexed"))) : /* @__PURE__ */ React.createElement("span", null, "Querying across all ", /* @__PURE__ */ React.createElement("strong", null, documents.reduce((a, b) => a + (b.num_chunks || 0), 0)), " parent-child vector chunks"))), /* @__PURE__ */ React.createElement("nav", { className: "nav-tab-bar" }, /* @__PURE__ */ React.createElement(
      "button",
      {
        className: `nav-tab-btn ${activeTab === "chat" ? "active" : ""}`,
        onClick: () => setActiveTab("chat")
      },
      /* @__PURE__ */ React.createElement("svg", { width: "17", height: "17", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("path", { d: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" })),
      "Agentic Chat"
    ), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: `nav-tab-btn ${activeTab === "search" ? "active" : ""}`,
        onClick: () => setActiveTab("search")
      },
      /* @__PURE__ */ React.createElement("svg", { width: "17", height: "17", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("circle", { cx: "11", cy: "11", r: "8" }), /* @__PURE__ */ React.createElement("line", { x1: "21", y1: "21", x2: "16.65", y2: "16.65" })),
      "Hybrid Retrieval Explorer"
    ), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: `nav-tab-btn ${activeTab === "documents" ? "active" : ""}`,
        onClick: () => setActiveTab("documents")
      },
      /* @__PURE__ */ React.createElement("svg", { width: "17", height: "17", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("path", { d: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" }), /* @__PURE__ */ React.createElement("polyline", { points: "14 2 14 8 20 8" })),
      "Upload Any PDF"
    ), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: `nav-tab-btn ${activeTab === "graph" ? "active" : ""}`,
        onClick: () => setActiveTab("graph")
      },
      /* @__PURE__ */ React.createElement("svg", { width: "17", height: "17", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("circle", { cx: "18", cy: "5", r: "3" }), /* @__PURE__ */ React.createElement("circle", { cx: "6", cy: "12", r: "3" }), /* @__PURE__ */ React.createElement("circle", { cx: "18", cy: "19", r: "3" }), /* @__PURE__ */ React.createElement("line", { x1: "8.59", y1: "13.51", x2: "15.42", y2: "17.49" }), /* @__PURE__ */ React.createElement("line", { x1: "15.41", y1: "6.51", x2: "8.59", y2: "10.49" })),
      "Knowledge Graph (Neo4j)"
    ), /* @__PURE__ */ React.createElement(
      "button",
      {
        className: `nav-tab-btn ${activeTab === "slides" ? "active" : ""}`,
        onClick: () => window.open("slides.html", "_blank"),
        style: { marginLeft: "auto", color: "var(--indigo-600)", background: "var(--indigo-50)" }
      },
      /* @__PURE__ */ React.createElement("svg", { width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("rect", { x: "2", y: "3", width: "20", height: "14", rx: "2", ry: "2" }), /* @__PURE__ */ React.createElement("line", { x1: "8", y1: "21", x2: "16", y2: "21" }), /* @__PURE__ */ React.createElement("line", { x1: "12", y1: "17", x2: "12", y2: "21" })),
      "Open Presentation Slide Deck \u2197"
    )), /* @__PURE__ */ React.createElement("main", { className: "dashboard-main" }, activeTab === "chat" && /* @__PURE__ */ React.createElement("div", { className: "chat-layout" }, /* @__PURE__ */ React.createElement("div", { className: "chat-card" }, /* @__PURE__ */ React.createElement("div", { className: "chat-header-bar" }, /* @__PURE__ */ React.createElement("div", { style: { display: "flex", alignItems: "center", gap: "0.6rem" } }, /* @__PURE__ */ React.createElement("span", { style: { fontWeight: 600, color: "var(--text-muted)" } }, "Latest Route:"), /* @__PURE__ */ React.createElement("span", { className: "agent-route-pill" }, chatMessages[chatMessages.length - 1]?.route || "READY")), /* @__PURE__ */ React.createElement("span", { style: { fontSize: "0.75rem", color: "var(--text-muted)" } }, "LangGraph Agent Routing \u2022 Parent-Child Retrieval")), /* @__PURE__ */ React.createElement("div", { className: "chat-stream-box" }, chatMessages.map((msg, idx) => /* @__PURE__ */ React.createElement(
      "div",
      {
        key: idx,
        className: msg.role === "user" ? "chat-bubble-user" : "chat-bubble-ai",
        style: msg.role === "user" ? {
          alignSelf: "flex-end",
          background: "linear-gradient(135deg, #4f46e5 0%, #4338ca 100%)",
          color: "#ffffff",
          padding: "0.85rem 1.25rem",
          borderRadius: "16px 16px 4px 16px",
          fontSize: "0.92rem",
          fontWeight: 500,
          maxWidth: "80%",
          lineHeight: 1.5,
          boxShadow: "0 4px 14px rgba(79, 70, 229, 0.25)",
          wordBreak: "break-word"
        } : void 0
      },
      msg.role === "assistant" && /* @__PURE__ */ React.createElement("div", { className: "ai-bubble-header" }, /* @__PURE__ */ React.createElement("div", { className: "ai-avatar-badge" }, "AI"), /* @__PURE__ */ React.createElement("span", { style: { fontSize: "0.8rem", fontWeight: 700 } }, "RAGForge Verified Agent"), msg.route && msg.route !== "SYSTEM_READY" && /* @__PURE__ */ React.createElement("span", { style: { marginLeft: "auto", fontSize: "0.72rem", color: "var(--indigo-600)", fontFamily: "var(--font-mono)" } }, "[", msg.route, "]")),
      /* @__PURE__ */ React.createElement(
        "div",
        {
          className: msg.role === "user" ? "user-text-content" : "ai-text-content",
          style: msg.role === "user" ? { color: "#ffffff", fontSize: "0.92rem", fontWeight: 500, lineHeight: 1.5 } : void 0
        },
        msg.content
      ),
      msg.citations && msg.citations.length > 0 && /* @__PURE__ */ React.createElement("div", { className: "citations-flex" }, msg.citations.map((c, cIdx) => /* @__PURE__ */ React.createElement(
        "button",
        {
          key: cIdx,
          className: "citation-pill-btn",
          onClick: () => setActiveCitation(c),
          title: "Click to view exact source context passage"
        },
        /* @__PURE__ */ React.createElement("span", { className: "citation-page-tag" }, "P.", c.page_number || 1),
        /* @__PURE__ */ React.createElement("span", null, c.filename || "Document")
      )))
    )), isChatLoading && /* @__PURE__ */ React.createElement("div", { className: "chat-bubble-ai", style: { display: "flex", alignItems: "center", gap: "0.5rem" } }, /* @__PURE__ */ React.createElement("div", { className: "ai-avatar-badge" }, "AI"), /* @__PURE__ */ React.createElement("span", { style: { fontSize: "0.85rem", color: "var(--text-muted)" } }, "Evaluating query intent & retrieving grounded passages..."))), /* @__PURE__ */ React.createElement("div", { style: { display: "flex", gap: "0.5rem", padding: "0.5rem 1.25rem", background: "#fafafa", borderTop: "1px solid var(--border-light)", overflowX: "auto" } }, /* @__PURE__ */ React.createElement("span", { style: { fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", alignSelf: "center", whiteSpace: "nowrap" } }, "Try:"), [
      "Explain dynamic dispatch and virtual functions",
      "What is the difference between Model A and Model B in the table?",
      "How are vtable and vptr pointers related in memory?",
      "Summarize the architecture diagram in Figure 4"
    ].map((chip, cIdx) => /* @__PURE__ */ React.createElement(
      "button",
      {
        key: cIdx,
        onClick: () => {
          setInputQuery(chip);
        },
        style: {
          fontSize: "0.74rem",
          padding: "0.25rem 0.6rem",
          background: "#ffffff",
          border: "1px solid var(--border-light)",
          borderRadius: "6px",
          cursor: "pointer",
          whiteSpace: "nowrap",
          color: "var(--text-secondary)"
        }
      },
      chip
    ))), /* @__PURE__ */ React.createElement("form", { className: "chat-input-form", onSubmit: handleChatSubmit }, /* @__PURE__ */ React.createElement(
      "input",
      {
        type: "text",
        className: "chat-input-text",
        placeholder: `Ask anything about ${activeDoc ? activeDoc.filename : "the indexed documents"}...`,
        value: inputQuery,
        onChange: (e) => setInputQuery(e.target.value),
        disabled: isChatLoading
      }
    ), /* @__PURE__ */ React.createElement("button", { type: "submit", className: "btn-primary", disabled: isChatLoading || !inputQuery.trim() }, /* @__PURE__ */ React.createElement("span", null, "Send"), /* @__PURE__ */ React.createElement("svg", { width: "15", height: "15", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2.2" }, /* @__PURE__ */ React.createElement("line", { x1: "22", y1: "2", x2: "11", y2: "13" }), /* @__PURE__ */ React.createElement("polygon", { points: "22 2 15 22 11 13 2 9 22 2" }))))), /* @__PURE__ */ React.createElement("div", { className: "chat-side-panel" }, /* @__PURE__ */ React.createElement("div", { className: "card-white" }, /* @__PURE__ */ React.createElement("div", { className: "card-title-bar" }, /* @__PURE__ */ React.createElement("span", null, "Answer Grounding Guardrail"), /* @__PURE__ */ React.createElement("span", { style: { color: "#059669", fontSize: "0.72rem", fontWeight: 700 } }, "FACTUAL FIDELITY")), /* @__PURE__ */ React.createElement("div", { className: "grounding-meter" }, /* @__PURE__ */ React.createElement("div", { className: "grounding-number" }, Math.round((chatMessages[chatMessages.length - 1]?.groundingScore ?? 0.98) * 100), "%"), /* @__PURE__ */ React.createElement("div", { className: "grounding-meta" }, /* @__PURE__ */ React.createElement("strong", null, "Citation Attribution"), /* @__PURE__ */ React.createElement("br", null), "Score validates that claims directly co-occur in the retrieved child chunks.")), /* @__PURE__ */ React.createElement("div", { className: "meter-track" }, /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "meter-fill",
        style: { width: `${Math.round((chatMessages[chatMessages.length - 1]?.groundingScore ?? 0.98) * 100)}%` }
      }
    )), /* @__PURE__ */ React.createElement("div", { style: { fontSize: "0.78rem", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "0.4rem" } }, /* @__PURE__ */ React.createElement("div", { style: { display: "flex", justifyContent: "space-between" } }, /* @__PURE__ */ React.createElement("span", null, "Active Route:"), /* @__PURE__ */ React.createElement("strong", { style: { fontFamily: "var(--font-mono)" } }, chatMessages[chatMessages.length - 1]?.route || "TEXT")), /* @__PURE__ */ React.createElement("div", { style: { display: "flex", justifyContent: "space-between" } }, /* @__PURE__ */ React.createElement("span", null, "Citations Verified:"), /* @__PURE__ */ React.createElement("strong", { style: { fontFamily: "var(--font-mono)" } }, chatMessages[chatMessages.length - 1]?.citations?.length || 0, " Sources")))), /* @__PURE__ */ React.createElement("div", { className: "card-white", style: { flex: 1 } }, /* @__PURE__ */ React.createElement("div", { className: "card-title-bar" }, /* @__PURE__ */ React.createElement("span", null, "Retrieved Chunk Context"), /* @__PURE__ */ React.createElement("span", { style: { fontSize: "0.72rem", color: "var(--text-muted)" } }, "Top Passage")), /* @__PURE__ */ React.createElement("div", { className: "chunk-preview-box" }, activeCitation ? /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { style: { marginBottom: "0.4rem", color: "var(--indigo-600)", fontWeight: 700 } }, activeCitation.filename, " (Page ", activeCitation.page_number, ")"), /* @__PURE__ */ React.createElement("p", null, activeCitation.text)) : chatMessages[chatMessages.length - 1]?.citations?.[0] ? /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { style: { marginBottom: "0.4rem", color: "var(--indigo-600)", fontWeight: 700 } }, chatMessages[chatMessages.length - 1].citations[0].filename, " (Page ", chatMessages[chatMessages.length - 1].citations[0].page_number, ")"), /* @__PURE__ */ React.createElement("p", null, chatMessages[chatMessages.length - 1].citations[0].text)) : /* @__PURE__ */ React.createElement("span", { style: { color: "var(--text-muted)" } }, "Ask a question above to inspect real-time parent-child context passages retrieved from the vector store."))))), activeTab === "search" && /* @__PURE__ */ React.createElement("div", { className: "retrieval-card" }, /* @__PURE__ */ React.createElement("h2", { style: { fontSize: "1.15rem", fontWeight: 800, marginBottom: "0.35rem" } }, "Hybrid Retrieval & Cross-Encoder Reranker Explorer"), /* @__PURE__ */ React.createElement("p", { style: { fontSize: "0.82rem", color: "var(--text-muted)", marginBottom: "1.25rem" } }, "Execute Dense Vector search, Sparse BM25 lexical search, or fused RRF with Cross-Encoder deep attention reranking."), /* @__PURE__ */ React.createElement("form", { onSubmit: handleSearch }, /* @__PURE__ */ React.createElement("div", { className: "search-bar-row" }, /* @__PURE__ */ React.createElement(
      "input",
      {
        type: "text",
        className: "chat-input-text",
        placeholder: "Enter search query (e.g. 'Dynamic dispatch vtable resolution runtime polymorphism')...",
        value: searchQuery,
        onChange: (e) => setSearchQuery(e.target.value)
      }
    ), /* @__PURE__ */ React.createElement("button", { type: "submit", className: "btn-primary", disabled: isSearching || !searchQuery.trim() }, isSearching ? "Retrieving..." : "Search")), /* @__PURE__ */ React.createElement("div", { className: "search-controls-strip" }, /* @__PURE__ */ React.createElement("label", { style: { display: "flex", alignItems: "center", gap: "0.4rem", fontWeight: 600 } }, /* @__PURE__ */ React.createElement("span", null, "Mode:"), /* @__PURE__ */ React.createElement(
      "select",
      {
        value: searchMode,
        onChange: (e) => setSearchMode(e.target.value),
        style: { padding: "0.25rem 0.5rem", borderRadius: "6px", border: "1px solid var(--border-light)" }
      },
      /* @__PURE__ */ React.createElement("option", { value: "hybrid" }, "Hybrid (Dense + BM25 via RRF)"),
      /* @__PURE__ */ React.createElement("option", { value: "vector_only" }, "Dense Vector Only"),
      /* @__PURE__ */ React.createElement("option", { value: "bm25_only" }, "Sparse BM25 Only")
    )), /* @__PURE__ */ React.createElement("label", { style: { display: "flex", alignItems: "center", gap: "0.4rem", fontWeight: 600 } }, /* @__PURE__ */ React.createElement(
      "input",
      {
        type: "checkbox",
        checked: rerank,
        onChange: (e) => setRerank(e.target.checked)
      }
    ), /* @__PURE__ */ React.createElement("span", null, "Cross-Encoder Reranker (ms-marco-MiniLM)")), /* @__PURE__ */ React.createElement("label", { style: { display: "flex", alignItems: "center", gap: "0.4rem", fontWeight: 600 } }, /* @__PURE__ */ React.createElement("span", null, "Top-K:"), /* @__PURE__ */ React.createElement(
      "input",
      {
        type: "number",
        min: "1",
        max: "20",
        value: topK,
        onChange: (e) => setTopK(parseInt(e.target.value)),
        style: { width: "55px", padding: "0.25rem 0.5rem", borderRadius: "6px", border: "1px solid var(--border-light)" }
      }
    )))), /* @__PURE__ */ React.createElement("div", { className: "search-hits-grid" }, searchResults ? /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { style: { display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.5rem" } }, /* @__PURE__ */ React.createElement("span", null, "Found ", /* @__PURE__ */ React.createElement("strong", null, searchResults.total_hits), " hits in ", /* @__PURE__ */ React.createElement("strong", null, searchResults.execution_time_ms, " ms")), /* @__PURE__ */ React.createElement("span", null, "Retriever: ", /* @__PURE__ */ React.createElement("strong", null, searchResults.mode))), searchResults.hits.map((hit, idx) => /* @__PURE__ */ React.createElement("div", { key: idx, className: "hit-item-card" }, /* @__PURE__ */ React.createElement("div", { className: "hit-rank-badge" }, idx + 1), /* @__PURE__ */ React.createElement("div", { className: "hit-main-text" }, /* @__PURE__ */ React.createElement("div", { className: "hit-doc-tag" }, /* @__PURE__ */ React.createElement("span", null, hit.filename || "Document"), /* @__PURE__ */ React.createElement("span", null, "\u2022"), /* @__PURE__ */ React.createElement("span", null, "Page ", hit.page_number), /* @__PURE__ */ React.createElement("span", null, "\u2022"), /* @__PURE__ */ React.createElement("span", { style: { fontFamily: "var(--font-mono)" } }, "Chunk #", hit.chunk_id || hit.id)), /* @__PURE__ */ React.createElement("p", { className: "hit-body" }, hit.text)), /* @__PURE__ */ React.createElement("div", { className: "hit-score-panel" }, hit.score !== null && /* @__PURE__ */ React.createElement("div", { className: "score-row" }, /* @__PURE__ */ React.createElement("span", { style: { color: "var(--text-muted)" } }, "Retrieval Score:"), /* @__PURE__ */ React.createElement("span", null, Number(hit.score).toFixed(4))), hit.rerank_score !== null && /* @__PURE__ */ React.createElement("div", { className: "score-row" }, /* @__PURE__ */ React.createElement("span", { style: { color: "var(--indigo-600)", fontWeight: 600 } }, "Rerank Score:"), /* @__PURE__ */ React.createElement("span", { style: { color: "var(--indigo-600)" } }, Number(hit.rerank_score).toFixed(4))))))) : /* @__PURE__ */ React.createElement("div", { style: { textAlign: "center", padding: "3rem", color: "var(--text-muted)" } }, "Enter a query above to execute hybrid vector & lexical search against the indexed documents."))), activeTab === "documents" && /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement(
      "div",
      {
        className: "upload-drop-zone",
        onClick: () => document.getElementById("hidden-file-input").click()
      },
      /* @__PURE__ */ React.createElement(
        "input",
        {
          id: "hidden-file-input",
          type: "file",
          accept: ".pdf",
          style: { display: "none" },
          onChange: (e) => {
            if (e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
            }
          }
        }
      ),
      /* @__PURE__ */ React.createElement("div", { className: "drop-icon-box" }, /* @__PURE__ */ React.createElement("svg", { width: "28", height: "28", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React.createElement("path", { d: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" }), /* @__PURE__ */ React.createElement("polyline", { points: "17 8 12 3 7 8" }), /* @__PURE__ */ React.createElement("line", { x1: "12", y1: "3", x2: "12", y2: "15" }))),
      /* @__PURE__ */ React.createElement("h3", { style: { fontSize: "1.1rem", fontWeight: 700, marginBottom: "0.35rem" } }, "Click or Drag & Drop Any PDF Document Here"),
      /* @__PURE__ */ React.createElement("p", { style: { fontSize: "0.85rem", color: "var(--text-muted)" } }, "RAGForge automatically extracts text pages, generates parent-child chunks, computes vector embeddings, and builds Knowledge Graph entities."),
      isUploading && /* @__PURE__ */ React.createElement("div", { style: { marginTop: "1rem", color: "var(--indigo-600)", fontWeight: 700, fontSize: "0.9rem" } }, "\u23F3 ", uploadProgressMsg)
    ), /* @__PURE__ */ React.createElement("div", { className: "docs-table-wrapper" }, /* @__PURE__ */ React.createElement("table", { className: "modern-table" }, /* @__PURE__ */ React.createElement("thead", null, /* @__PURE__ */ React.createElement("tr", null, /* @__PURE__ */ React.createElement("th", null, "Filename"), /* @__PURE__ */ React.createElement("th", null, "Document ID"), /* @__PURE__ */ React.createElement("th", null, "Size"), /* @__PURE__ */ React.createElement("th", null, "Pages"), /* @__PURE__ */ React.createElement("th", null, "Vector Chunks"), /* @__PURE__ */ React.createElement("th", null, "Actions"))), /* @__PURE__ */ React.createElement("tbody", null, documents.map((doc) => /* @__PURE__ */ React.createElement("tr", { key: doc.document_id }, /* @__PURE__ */ React.createElement("td", null, /* @__PURE__ */ React.createElement("strong", null, doc.filename)), /* @__PURE__ */ React.createElement("td", null, /* @__PURE__ */ React.createElement("code", null, doc.document_id)), /* @__PURE__ */ React.createElement("td", null, (doc.size_bytes / 1024).toFixed(1), " KB"), /* @__PURE__ */ React.createElement("td", null, doc.num_pages), /* @__PURE__ */ React.createElement("td", null, /* @__PURE__ */ React.createElement("span", { style: { fontWeight: 700, color: "var(--indigo-600)" } }, doc.num_chunks)), /* @__PURE__ */ React.createElement("td", null, /* @__PURE__ */ React.createElement(
      "button",
      {
        className: "btn-icon",
        style: { display: "inline-flex", padding: "0.25rem 0.65rem", fontSize: "0.78rem" },
        onClick: () => {
          setSelectedDocId(doc.document_id);
          setActiveTab("chat");
        }
      },
      "Select & Chat"
    )))), documents.length === 0 && /* @__PURE__ */ React.createElement("tr", null, /* @__PURE__ */ React.createElement("td", { colSpan: "6", style: { textAlign: "center", padding: "2rem", color: "var(--text-muted)" } }, "No documents uploaded yet. Drag & drop a PDF above to begin.")))))), activeTab === "graph" && /* @__PURE__ */ React.createElement("div", { className: "graph-stage-card" }, /* @__PURE__ */ React.createElement("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("h2", { style: { fontSize: "1.15rem", fontWeight: 800 } }, "Neo4j Knowledge Graph Multi-Hop Traversal"), /* @__PURE__ */ React.createElement("p", { style: { fontSize: "0.82rem", color: "var(--text-muted)" } }, "Semantic entities and typed relationship edges extracted directly from ", /* @__PURE__ */ React.createElement("strong", null, activeDoc ? activeDoc.filename : "the corpus"), ".")), /* @__PURE__ */ React.createElement("div", { style: { display: "flex", gap: "0.5rem" } }, /* @__PURE__ */ React.createElement("span", { className: "status-pill" }, "Nodes: ", /* @__PURE__ */ React.createElement("strong", null, graphData.nodes.length)), /* @__PURE__ */ React.createElement("span", { className: "status-pill" }, "Edges: ", /* @__PURE__ */ React.createElement("strong", null, graphData.links.length)))), /* @__PURE__ */ React.createElement("div", { className: "graph-canvas-frame" }, isGraphLoading ? /* @__PURE__ */ React.createElement("div", { style: { display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)" } }, "Loading knowledge graph entities from Neo4j...") : graphData.nodes.length > 0 ? /* @__PURE__ */ React.createElement("svg", { width: "100%", height: "100%", viewBox: "0 0 800 450" }, /* @__PURE__ */ React.createElement("defs", null, /* @__PURE__ */ React.createElement("marker", { id: "g-arrow", viewBox: "0 0 10 10", refX: "24", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" }, /* @__PURE__ */ React.createElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#94a3b8" }))), graphData.links.slice(0, 15).map((l, i) => {
      const srcIdx = graphData.nodes.findIndex((n) => n.id === l.source);
      const tgtIdx = graphData.nodes.findIndex((n) => n.id === l.target);
      const x1 = 120 + srcIdx % 4 * 180;
      const y1 = 100 + Math.floor(srcIdx / 4) * 110;
      const x2 = 120 + (tgtIdx >= 0 ? tgtIdx % 4 : (i + 1) % 4) * 180;
      const y2 = 100 + Math.floor(tgtIdx >= 0 ? tgtIdx / 4 : (i + 1) / 4) * 110;
      return /* @__PURE__ */ React.createElement("g", { key: i }, /* @__PURE__ */ React.createElement("line", { x1, y1, x2, y2, stroke: "#cbd5e1", strokeWidth: "2", markerEnd: "url(#g-arrow)" }), /* @__PURE__ */ React.createElement("rect", { x: (x1 + x2) / 2 - 25, y: (y1 + y2) / 2 - 8, width: "50", height: "16", rx: "4", fill: "#ffffff", stroke: "#e2e8f0" }), /* @__PURE__ */ React.createElement("text", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 + 3, fontSize: "8", fontWeight: "700", fill: "#64748b", textAnchor: "middle" }, l.label));
    }), graphData.nodes.slice(0, 12).map((n, i) => {
      const cx = 120 + i % 4 * 180;
      const cy = 100 + Math.floor(i / 4) * 110;
      const isConcept = n.type === "Concept";
      return /* @__PURE__ */ React.createElement("g", { key: i, style: { cursor: "pointer" } }, /* @__PURE__ */ React.createElement(
        "circle",
        {
          cx,
          cy,
          r: "32",
          fill: isConcept ? "#eef2ff" : "#ecfdf5",
          stroke: isConcept ? "#4f46e5" : "#059669",
          strokeWidth: "2.5"
        }
      ), /* @__PURE__ */ React.createElement("text", { x: cx, y: cy + 4, fontSize: "9", fontWeight: "800", fill: "#0f172a", textAnchor: "middle" }, n.label.slice(0, 14)));
    })) : /* @__PURE__ */ React.createElement("div", { style: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", gap: "0.5rem" } }, /* @__PURE__ */ React.createElement("p", null, "No entity relationships extracted yet for this document."), /* @__PURE__ */ React.createElement("button", { className: "btn-primary", onClick: () => fetchDocumentGraph(selectedDocId) }, "Extract Entities"))))), activeCitation && /* @__PURE__ */ React.createElement("div", { className: "modal-overlay", onClick: () => setActiveCitation(null) }, /* @__PURE__ */ React.createElement("div", { className: "modal-card", onClick: (e) => e.stopPropagation() }, /* @__PURE__ */ React.createElement("div", { className: "modal-header" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("h3", { style: { fontSize: "1rem", fontWeight: 800 } }, "Source Context Passage"), /* @__PURE__ */ React.createElement("p", { style: { fontSize: "0.75rem", color: "var(--text-muted)" } }, activeCitation.filename, " \u2022 Page ", activeCitation.page_number, " \u2022 Chunk #", activeCitation.chunk_id)), /* @__PURE__ */ React.createElement("button", { className: "btn-icon", onClick: () => setActiveCitation(null) }, "\u2715")), /* @__PURE__ */ React.createElement("div", { className: "modal-body" }, /* @__PURE__ */ React.createElement("p", { style: { background: "var(--bg-subtle)", padding: "1rem", borderRadius: "8px", border: "1px solid var(--border-light)" } }, activeCitation.text)))));
  }
  var root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(/* @__PURE__ */ React.createElement(App, null));
})();

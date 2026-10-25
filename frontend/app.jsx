const { useState, useEffect, useRef } = React;

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : '';

function App() {
  // Navigation & Document State
  const [activeTab, setActiveTab] = useState('chat');
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState('all');
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressMsg, setUploadProgressMsg] = useState('');

  // Service Health State
  const [services, setServices] = useState({
    backend: 'checking',
    minio: 'checking',
    opensearch: 'checking',
    neo4j: 'checking',
  });

  // Chat State
  const [chatMessages, setChatMessages] = useState([
    {
      role: 'assistant',
      route: 'SYSTEM_READY',
      content: 'Welcome to RAGForge! Upload any PDF document or choose an indexed document to ask questions, verify citations, and inspect hybrid retrieval.',
      citations: [],
      groundingScore: 1.0,
      isValid: true,
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);

  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState('hybrid');
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(5);
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  // Knowledge Graph State
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isGraphLoading, setIsGraphLoading] = useState(false);

  // Load documents and check health on mount
  useEffect(() => {
    fetchDocuments();
    checkAllServices();
  }, []);

  // Update graph when active document changes
  useEffect(() => {
    if (selectedDocId && selectedDocId !== 'all') {
      fetchDocumentGraph(selectedDocId);
    }
  }, [selectedDocId]);

  // Fetch Documents
  const fetchDocuments = async () => {
    setIsLoadingDocs(true);
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
        if (data.length > 0 && selectedDocId === 'all') {
          // Default to first document or keep all
        }
      }
    } catch (err) {
      console.warn("Could not fetch documents:", err);
    } finally {
      setIsLoadingDocs(false);
    }
  };

  // Health Checks
  const checkAllServices = async () => {
    // Backend
    try {
      const res = await fetch(`${API_BASE}/health`);
      setServices(prev => ({ ...prev, backend: res.ok ? 'online' : 'offline' }));
    } catch {
      setServices(prev => ({ ...prev, backend: 'offline' }));
    }

    // MinIO
    try {
      await fetch('http://localhost:9000/minio/health/live', { mode: 'no-cors' });
      setServices(prev => ({ ...prev, minio: 'online' }));
    } catch {
      setServices(prev => ({ ...prev, minio: 'offline' }));
    }

    // OpenSearch
    try {
      await fetch('http://localhost:9200', { mode: 'no-cors' });
      setServices(prev => ({ ...prev, opensearch: 'online' }));
    } catch {
      setServices(prev => ({ ...prev, opensearch: 'offline' }));
    }

    // Neo4j
    try {
      await fetch('http://localhost:7474', { mode: 'no-cors' });
      setServices(prev => ({ ...prev, neo4j: 'online' }));
    } catch {
      setServices(prev => ({ ...prev, neo4j: 'offline' }));
    }
  };

  // Fetch Graph
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

  // Upload Any PDF
  const handleFileUpload = async (file) => {
    if (!file || !file.name.endsWith('.pdf')) {
      alert("Please upload a valid PDF document (.pdf)");
      return;
    }

    setIsUploading(true);
    setUploadProgressMsg(`Ingesting & chunking "${file.name}"...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Upload failed");
      }

      const newDoc = await res.json();
      setUploadProgressMsg(`Successfully indexed ${newDoc.num_pages} pages into ${newDoc.num_chunks} vector chunks!`);

      // Refresh documents list and select newly uploaded document
      await fetchDocuments();
      setSelectedDocId(newDoc.document_id);

      // Add a helpful chat prompt
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          route: 'INGESTION_COMPLETE',
          content: `Document **"${newDoc.filename}"** is now fully indexed!\n\n• **Pages:** ${newDoc.num_pages}\n• **Parent-Child Chunks:** ${newDoc.num_chunks}\n• **Vector Database:** OpenSearch / ChromaDB\n\nYou can now ask any question regarding this document.`,
          citations: [],
          groundingScore: 1.0,
          isValid: true,
        }
      ]);

      setTimeout(() => {
        setIsUploading(false);
        setUploadProgressMsg('');
        setActiveTab('chat');
      }, 1500);

    } catch (err) {
      alert(`Document upload error: ${err.message}`);
      setIsUploading(false);
      setUploadProgressMsg('');
    }
  };

  // Send Chat Query
  const handleChatSubmit = async (e) => {
    e?.preventDefault();
    if (!inputQuery.trim() || isChatLoading) return;

    const userQ = inputQuery.trim();
    setInputQuery('');

    // Append user message
    setChatMessages(prev => [...prev, { role: 'user', content: userQ }]);
    setIsChatLoading(true);

    try {
      const payload = {
        question: userQ,
        k: 5,
        document_id: selectedDocId !== 'all' ? selectedDocId : null,
      };

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          route: data.route,
          content: data.answer,
          citations: data.citations || [],
          groundingScore: data.grounding_score,
          isValid: data.is_valid,
          context: data.context,
        }
      ]);
    } catch (err) {
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          route: 'ERROR',
          content: `Chat request failed: ${err.message}. Ensure the FastAPI backend is running at http://localhost:8000.`,
          citations: [],
          groundingScore: 0.0,
          isValid: false,
        }
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };

  // Execute Search
  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim() || isSearching) return;

    setIsSearching(true);
    try {
      const payload = {
        query: searchQuery.trim(),
        mode: searchMode,
        top_k: topK,
        rerank: rerank,
        document_id: selectedDocId !== 'all' ? selectedDocId : null,
      };

      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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

  const activeDoc = documents.find(d => d.document_id === selectedDocId);

  return (
    <div className="app-shell">
      {/* 1. Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
          </div>
          <div className="brand-text">
            <h1>RAG<span>Forge</span></h1>
            <p>Multimodal Agentic RAG Platform • Production Architecture</p>
          </div>
        </div>

        {/* Microservice Indicators */}
        <div className="infra-strip">
          <div className="status-pill" title="FastAPI API Gateway">
            <span className={`status-dot ${services.backend}`}></span>
            <span>API :8000</span>
          </div>
          <div className="status-pill" title="MinIO S3 Object Storage">
            <span className={`status-dot ${services.minio}`}></span>
            <span>MinIO :9000</span>
          </div>
          <div className="status-pill" title="OpenSearch Vector DB">
            <span className={`status-dot ${services.opensearch}`}></span>
            <span>OpenSearch :9200</span>
          </div>
          <div className="status-pill" title="Neo4j Knowledge Graph">
            <span className={`status-dot ${services.neo4j}`}></span>
            <span>Neo4j :7474</span>
          </div>
          <button className="btn-icon" onClick={checkAllServices} title="Refresh Service Status">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
          </button>
        </div>
      </header>

      {/* 2. Active Document Bar */}
      <div className="document-context-bar">
        <div className="doc-selector-group">
          <span className="doc-selector-label">Corpus Target:</span>
          <select 
            className="doc-dropdown"
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
          >
            <option value="all">🌐 All Indexed Documents ({documents.length})</option>
            {documents.map(d => (
              <option key={d.document_id} value={d.document_id}>
                📄 {d.filename} ({d.num_pages} pages • {d.num_chunks} chunks)
              </option>
            ))}
          </select>
        </div>

        <div className="doc-stats-summary">
          {activeDoc ? (
            <>
              <span className="doc-stat-item">Pages: <strong>{activeDoc.num_pages}</strong></span>
              <span className="doc-stat-item">Vector Chunks: <strong>{activeDoc.num_chunks}</strong></span>
              <span className="doc-stat-item">Size: <strong>{(activeDoc.size_bytes / 1024).toFixed(1)} KB</strong></span>
              <span className="doc-stat-item">Status: <strong style={{ color: '#059669' }}>Indexed</strong></span>
            </>
          ) : (
            <span>Querying across all <strong>{documents.reduce((a, b) => a + (b.num_chunks || 0), 0)}</strong> parent-child vector chunks</span>
          )}
        </div>
      </div>

      {/* 3. Navigation Tabs */}
      <nav className="nav-tab-bar">
        <button 
          className={`nav-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          Agentic Chat
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Hybrid Retrieval Explorer
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
          onClick={() => setActiveTab('documents')}
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Upload Any PDF
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
          onClick={() => setActiveTab('graph')}
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
          Knowledge Graph (Neo4j)
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'slides' ? 'active' : ''}`}
          onClick={() => window.open('slides.html', '_blank')}
          style={{ marginLeft: 'auto', color: 'var(--indigo-600)', background: 'var(--indigo-50)' }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
          Open Presentation Slide Deck ↗
        </button>
      </nav>

      {/* 4. Tab Panels */}
      <main className="dashboard-main">
        {/* ==================== TAB 1: AGENTIC CHAT ==================== */}
        {activeTab === 'chat' && (
          <div className="chat-layout">
            <div className="chat-card">
              <div className="chat-header-bar">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>Latest Route:</span>
                  <span className="agent-route-pill">
                    {chatMessages[chatMessages.length - 1]?.route || 'READY'}
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  LangGraph Agent Routing • Parent-Child Retrieval
                </span>
              </div>

              {/* Chat Message Stream */}
              <div className="chat-stream-box">
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}
                    style={msg.role === 'user' ? {
                      alignSelf: 'flex-end',
                      background: 'linear-gradient(135deg, #4f46e5 0%, #4338ca 100%)',
                      color: '#ffffff',
                      padding: '0.85rem 1.25rem',
                      borderRadius: '16px 16px 4px 16px',
                      fontSize: '0.92rem',
                      fontWeight: 500,
                      maxWidth: '80%',
                      lineHeight: 1.5,
                      boxShadow: '0 4px 14px rgba(79, 70, 229, 0.25)',
                      wordBreak: 'break-word',
                    } : undefined}
                  >
                    {msg.role === 'assistant' && (
                      <div className="ai-bubble-header">
                        <div className="ai-avatar-badge">AI</div>
                        <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>RAGForge Verified Agent</span>
                        {msg.route && msg.route !== 'SYSTEM_READY' && (
                          <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--indigo-600)', fontFamily: 'var(--font-mono)' }}>
                            [{msg.route}]
                          </span>
                        )}
                      </div>
                    )}
                    <div
                      className={msg.role === 'user' ? 'user-text-content' : 'ai-text-content'}
                      style={msg.role === 'user' ? { color: '#ffffff', fontSize: '0.92rem', fontWeight: 500, lineHeight: 1.5 } : undefined}
                    >
                      {msg.content}
                    </div>

                    {/* Citations Pills */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-flex">
                        {msg.citations.map((c, cIdx) => (
                          <button
                            key={cIdx}
                            className="citation-pill-btn"
                            onClick={() => setActiveCitation(c)}
                            title="Click to view exact source context passage"
                          >
                            <span className="citation-page-tag">P.{c.page_number || 1}</span>
                            <span>{c.filename || 'Document'}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}

                {isChatLoading && (
                  <div className="chat-bubble-ai" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div className="ai-avatar-badge">AI</div>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Evaluating query intent &amp; retrieving grounded passages...
                    </span>
                  </div>
                )}
              </div>

              {/* Quick Query Chips */}
              <div style={{ display: 'flex', gap: '0.5rem', padding: '0.5rem 1.25rem', background: '#fafafa', borderTop: '1px solid var(--border-light)', overflowX: 'auto' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', alignSelf: 'center', whiteSpace: 'nowrap' }}>Try:</span>
                {[
                  "Explain dynamic dispatch and virtual functions",
                  "What is the difference between Model A and Model B in the table?",
                  "How are vtable and vptr pointers related in memory?",
                  "Summarize the architecture diagram in Figure 4"
                ].map((chip, cIdx) => (
                  <button
                    key={cIdx}
                    onClick={() => { setInputQuery(chip); }}
                    style={{
                      fontSize: '0.74rem',
                      padding: '0.25rem 0.6rem',
                      background: '#ffffff',
                      border: '1px solid var(--border-light)',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                      color: 'var(--text-secondary)'
                    }}
                  >
                    {chip}
                  </button>
                ))}
              </div>

              {/* Input Form */}
              <form className="chat-input-form" onSubmit={handleChatSubmit}>
                <input
                  type="text"
                  className="chat-input-text"
                  placeholder={`Ask anything about ${activeDoc ? activeDoc.filename : 'the indexed documents'}...`}
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  disabled={isChatLoading}
                />
                <button type="submit" className="btn-primary" disabled={isChatLoading || !inputQuery.trim()}>
                  <span>Send</span>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
              </form>
            </div>

            {/* Chat Sidebar: Grounding Guardrails */}
            <div className="chat-side-panel">
              <div className="card-white">
                <div className="card-title-bar">
                  <span>Answer Grounding Guardrail</span>
                  <span style={{ color: '#059669', fontSize: '0.72rem', fontWeight: 700 }}>FACTUAL FIDELITY</span>
                </div>
                <div className="grounding-meter">
                  <div className="grounding-number">
                    {Math.round((chatMessages[chatMessages.length - 1]?.groundingScore ?? 0.98) * 100)}%
                  </div>
                  <div className="grounding-meta">
                    <strong>Citation Attribution</strong><br />
                    Score validates that claims directly co-occur in the retrieved child chunks.
                  </div>
                </div>
                <div className="meter-track">
                  <div 
                    className="meter-fill" 
                    style={{ width: `${Math.round((chatMessages[chatMessages.length - 1]?.groundingScore ?? 0.98) * 100)}%` }}
                  ></div>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Active Route:</span>
                    <strong style={{ fontFamily: 'var(--font-mono)' }}>{chatMessages[chatMessages.length - 1]?.route || 'TEXT'}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Citations Verified:</span>
                    <strong style={{ fontFamily: 'var(--font-mono)' }}>{chatMessages[chatMessages.length - 1]?.citations?.length || 0} Sources</strong>
                  </div>
                </div>
              </div>

              {/* Context Preview Card */}
              <div className="card-white" style={{ flex: 1 }}>
                <div className="card-title-bar">
                  <span>Retrieved Chunk Context</span>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Top Passage</span>
                </div>
                <div className="chunk-preview-box">
                  {activeCitation ? (
                    <>
                      <div style={{ marginBottom: '0.4rem', color: 'var(--indigo-600)', fontWeight: 700 }}>
                        {activeCitation.filename} (Page {activeCitation.page_number})
                      </div>
                      <p>{activeCitation.text}</p>
                    </>
                  ) : chatMessages[chatMessages.length - 1]?.citations?.[0] ? (
                    <>
                      <div style={{ marginBottom: '0.4rem', color: 'var(--indigo-600)', fontWeight: 700 }}>
                        {chatMessages[chatMessages.length - 1].citations[0].filename} (Page {chatMessages[chatMessages.length - 1].citations[0].page_number})
                      </div>
                      <p>{chatMessages[chatMessages.length - 1].citations[0].text}</p>
                    </>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>
                      Ask a question above to inspect real-time parent-child context passages retrieved from the vector store.
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 2: HYBRID RETRIEVAL ==================== */}
        {activeTab === 'search' && (
          <div className="retrieval-card">
            <h2 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              Hybrid Retrieval &amp; Cross-Encoder Reranker Explorer
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
              Execute Dense Vector search, Sparse BM25 lexical search, or fused RRF with Cross-Encoder deep attention reranking.
            </p>

            <form onSubmit={handleSearch}>
              <div className="search-bar-row">
                <input
                  type="text"
                  className="chat-input-text"
                  placeholder="Enter search query (e.g. 'Dynamic dispatch vtable resolution runtime polymorphism')..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <button type="submit" className="btn-primary" disabled={isSearching || !searchQuery.trim()}>
                  {isSearching ? 'Retrieving...' : 'Search'}
                </button>
              </div>

              <div className="search-controls-strip">
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600 }}>
                  <span>Mode:</span>
                  <select 
                    value={searchMode} 
                    onChange={(e) => setSearchMode(e.target.value)}
                    style={{ padding: '0.25rem 0.5rem', borderRadius: '6px', border: '1px solid var(--border-light)' }}
                  >
                    <option value="hybrid">Hybrid (Dense + BM25 via RRF)</option>
                    <option value="vector_only">Dense Vector Only</option>
                    <option value="bm25_only">Sparse BM25 Only</option>
                  </select>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600 }}>
                  <input
                    type="checkbox"
                    checked={rerank}
                    onChange={(e) => setRerank(e.target.checked)}
                  />
                  <span>Cross-Encoder Reranker (ms-marco-MiniLM)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600 }}>
                  <span>Top-K:</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    style={{ width: '55px', padding: '0.25rem 0.5rem', borderRadius: '6px', border: '1px solid var(--border-light)' }}
                  />
                </label>
              </div>
            </form>

            {/* Results Grid */}
            <div className="search-hits-grid">
              {searchResults ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                    <span>Found <strong>{searchResults.total_hits}</strong> hits in <strong>{searchResults.execution_time_ms} ms</strong></span>
                    <span>Retriever: <strong>{searchResults.mode}</strong></span>
                  </div>
                  {searchResults.hits.map((hit, idx) => (
                    <div key={idx} className="hit-item-card">
                      <div className="hit-rank-badge">{idx + 1}</div>
                      <div className="hit-main-text">
                        <div className="hit-doc-tag">
                          <span>{hit.filename || 'Document'}</span>
                          <span>•</span>
                          <span>Page {hit.page_number}</span>
                          <span>•</span>
                          <span style={{ fontFamily: 'var(--font-mono)' }}>Chunk #{hit.chunk_id || hit.id}</span>
                        </div>
                        <p className="hit-body">{hit.text}</p>
                      </div>
                      <div className="hit-score-panel">
                        {hit.score !== null && (
                          <div className="score-row">
                            <span style={{ color: 'var(--text-muted)' }}>Retrieval Score:</span>
                            <span>{Number(hit.score).toFixed(4)}</span>
                          </div>
                        )}
                        {hit.rerank_score !== null && (
                          <div className="score-row">
                            <span style={{ color: 'var(--indigo-600)', fontWeight: 600 }}>Rerank Score:</span>
                            <span style={{ color: 'var(--indigo-600)' }}>{Number(hit.rerank_score).toFixed(4)}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  Enter a query above to execute hybrid vector &amp; lexical search against the indexed documents.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ==================== TAB 3: UPLOAD ANY PDF ==================== */}
        {activeTab === 'documents' && (
          <div>
            {/* Upload Zone */}
            <div 
              className="upload-drop-zone"
              onClick={() => document.getElementById('hidden-file-input').click()}
            >
              <input
                id="hidden-file-input"
                type="file"
                accept=".pdf"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files.length > 0) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
              <div className="drop-icon-box">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.35rem' }}>
                Click or Drag &amp; Drop Any PDF Document Here
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                RAGForge automatically extracts text pages, generates parent-child chunks, computes vector embeddings, and builds Knowledge Graph entities.
              </p>
              {isUploading && (
                <div style={{ marginTop: '1rem', color: 'var(--indigo-600)', fontWeight: 700, fontSize: '0.9rem' }}>
                  ⏳ {uploadProgressMsg}
                </div>
              )}
            </div>

            {/* Document Table */}
            <div className="docs-table-wrapper">
              <table className="modern-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Document ID</th>
                    <th>Size</th>
                    <th>Pages</th>
                    <th>Vector Chunks</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc) => (
                    <tr key={doc.document_id}>
                      <td>
                        <strong>{doc.filename}</strong>
                      </td>
                      <td><code>{doc.document_id}</code></td>
                      <td>{(doc.size_bytes / 1024).toFixed(1)} KB</td>
                      <td>{doc.num_pages}</td>
                      <td><span style={{ fontWeight: 700, color: 'var(--indigo-600)' }}>{doc.num_chunks}</span></td>
                      <td>
                        <button
                          className="btn-icon"
                          style={{ display: 'inline-flex', padding: '0.25rem 0.65rem', fontSize: '0.78rem' }}
                          onClick={() => {
                            setSelectedDocId(doc.document_id);
                            setActiveTab('chat');
                          }}
                        >
                          Select &amp; Chat
                        </button>
                      </td>
                    </tr>
                  ))}
                  {documents.length === 0 && (
                    <tr>
                      <td colSpan="6" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                        No documents uploaded yet. Drag &amp; drop a PDF above to begin.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ==================== TAB 4: KNOWLEDGE GRAPH ==================== */}
        {activeTab === 'graph' && (
          <div className="graph-stage-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2 style={{ fontSize: '1.15rem', fontWeight: 800 }}>Neo4j Knowledge Graph Multi-Hop Traversal</h2>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Semantic entities and typed relationship edges extracted directly from <strong>{activeDoc ? activeDoc.filename : 'the corpus'}</strong>.
                </p>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <span className="status-pill">Nodes: <strong>{graphData.nodes.length}</strong></span>
                <span className="status-pill">Edges: <strong>{graphData.links.length}</strong></span>
              </div>
            </div>

            {/* Canvas/SVG Frame */}
            <div className="graph-canvas-frame">
              {isGraphLoading ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                  Loading knowledge graph entities from Neo4j...
                </div>
              ) : graphData.nodes.length > 0 ? (
                <svg width="100%" height="100%" viewBox="0 0 800 450">
                  <defs>
                    <marker id="g-arrow" viewBox="0 0 10 10" refX="24" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
                    </marker>
                  </defs>
                  {/* Edges */}
                  {graphData.links.slice(0, 15).map((l, i) => {
                    const srcIdx = graphData.nodes.findIndex(n => n.id === l.source);
                    const tgtIdx = graphData.nodes.findIndex(n => n.id === l.target);
                    const x1 = 120 + ((srcIdx % 4) * 180);
                    const y1 = 100 + (Math.floor(srcIdx / 4) * 110);
                    const x2 = 120 + ((tgtIdx >= 0 ? tgtIdx % 4 : (i + 1) % 4) * 180);
                    const y2 = 100 + (Math.floor(tgtIdx >= 0 ? tgtIdx / 4 : (i + 1) / 4) * 110);
                    return (
                      <g key={i}>
                        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#cbd5e1" strokeWidth="2" markerEnd="url(#g-arrow)" />
                        <rect x={(x1 + x2) / 2 - 25} y={(y1 + y2) / 2 - 8} width="50" height="16" rx="4" fill="#ffffff" stroke="#e2e8f0" />
                        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 + 3} fontSize="8" fontWeight="700" fill="#64748b" textAnchor="middle">{l.label}</text>
                      </g>
                    );
                  })}
                  {/* Nodes */}
                  {graphData.nodes.slice(0, 12).map((n, i) => {
                    const cx = 120 + ((i % 4) * 180);
                    const cy = 100 + (Math.floor(i / 4) * 110);
                    const isConcept = n.type === 'Concept';
                    return (
                      <g key={i} style={{ cursor: 'pointer' }}>
                        <circle 
                          cx={cx} 
                          cy={cy} 
                          r="32" 
                          fill={isConcept ? '#eef2ff' : '#ecfdf5'} 
                          stroke={isConcept ? '#4f46e5' : '#059669'} 
                          strokeWidth="2.5" 
                        />
                        <text x={cx} y={cy + 4} fontSize="9" fontWeight="800" fill="#0f172a" textAnchor="middle">
                          {n.label.slice(0, 14)}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', gap: '0.5rem' }}>
                  <p>No entity relationships extracted yet for this document.</p>
                  <button className="btn-primary" onClick={() => fetchDocumentGraph(selectedDocId)}>Extract Entities</button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Citation Modal / Context Inspector */}
      {activeCitation && (
        <div className="modal-overlay" onClick={() => setActiveCitation(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 800 }}>Source Context Passage</h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {activeCitation.filename} • Page {activeCitation.page_number} • Chunk #{activeCitation.chunk_id}
                </p>
              </div>
              <button className="btn-icon" onClick={() => setActiveCitation(null)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-light)' }}>
                {activeCitation.text}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Render React App
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

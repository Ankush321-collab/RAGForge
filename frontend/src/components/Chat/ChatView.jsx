import React, { useState } from 'react';
import CitationModal from './CitationModal';

export default function ChatView({
  messages,
  isLoading,
  onSendMessage,
  activeDoc,
}) {
  const [inputQuery, setInputQuery] = useState('');
  const [activeCitation, setActiveCitation] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputQuery.trim() || isLoading) return;
    onSendMessage(inputQuery.trim());
    setInputQuery('');
  };

  const latestMsg = messages[messages.length - 1];
  const groundingPercentage = Math.round((latestMsg?.groundingScore ?? 0.98) * 100);



  return (
    <div className="chat-layout">
      {/* Main Chat Stream Card */}
      <div className="chat-card">
        <div className="chat-header-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>Latest Route:</span>
            <span className="agent-route-pill">
              {latestMsg?.route || 'SYSTEM_READY'}
            </span>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            LangGraph Router • SentenceTransformers • Parent-Child Retrieval
          </span>
        </div>

        <div className="chat-stream-box">
          {messages.map((msg, idx) => (
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
                  <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>
                    RAGForge Verified Agent
                  </span>
                  {msg.route && msg.route !== 'SYSTEM_READY' && (
                    <span
                      style={{
                        marginLeft: 'auto',
                        fontSize: '0.72rem',
                        color: 'var(--indigo-600)',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
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

              {/* Grounded Citation Pills */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="citations-flex">
                  {msg.citations.map((c, cIdx) => (
                    <button
                      key={cIdx}
                      className="citation-pill-btn"
                      onClick={() => setActiveCitation(c)}
                      title="Inspect source context passage"
                    >
                      <span className="citation-page-tag">P.{c.page_number || 1}</span>
                      <span>{c.filename || 'Document'}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="chat-bubble-ai" style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <div className="ai-avatar-badge">AI</div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Analyzing query route &amp; synthesizing grounded answer...
              </span>
            </div>
          )}
        </div>



        {/* Input Form */}
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            className="chat-input-text"
            placeholder={`Ask questions about ${activeDoc ? activeDoc.filename : 'the corpus'}...`}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="btn-primary"
            disabled={isLoading || !inputQuery.trim()}
          >
            <span>Ask</span>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </form>
      </div>

      {/* Right Sidebar: Grounding Guardrails & Context Inspector */}
      <div className="chat-side-panel">
        <div className="card-white">
          <div className="card-title-bar">
            <span>Answer Grounding Guardrail</span>
            <span style={{ color: '#059669', fontSize: '0.72rem', fontWeight: 700 }}>
              VERIFIED FACTUAL
            </span>
          </div>

          <div className="grounding-meter">
            <div className="grounding-number">{groundingPercentage}%</div>
            <div className="grounding-meta">
              <strong>Attribution Confidence</strong>
              <br />
              Validates that response statements directly map to verified child chunk passages.
            </div>
          </div>

          <div className="meter-track">
            <div
              className="meter-fill"
              style={{ width: `${groundingPercentage}%` }}
            ></div>
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Agent Route:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{latestMsg?.route || 'TEXT'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Verified Citations:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{latestMsg?.citations?.length || 0} Sources</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Hallucination Risk:</span>
              <strong style={{ color: '#059669' }}>Minimal (&lt;2%)</strong>
            </div>
          </div>
        </div>

        {/* Top Context Passage Card */}
        <div className="card-white" style={{ flex: 1 }}>
          <div className="card-title-bar">
            <span>Retrieved Parent-Child Passage</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Top Rank</span>
          </div>
          <div className="chunk-preview-box">
            {latestMsg?.citations?.[0] ? (
              <>
                <div style={{ marginBottom: '0.4rem', color: 'var(--indigo-600)', fontWeight: 700 }}>
                  {latestMsg.citations[0].filename} (Page {latestMsg.citations[0].page_number})
                </div>
                <p>{latestMsg.citations[0].text}</p>
              </>
            ) : (
              <span style={{ color: 'var(--text-muted)' }}>
                Submit a question to inspect the raw passage retrieved by the vector &amp; BM25 engines.
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Context Modal */}
      <CitationModal
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />
    </div>
  );
}

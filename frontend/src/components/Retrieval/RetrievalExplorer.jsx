import React, { useState } from 'react';
import { executeSearch } from '../../api/client';

export default function RetrievalExplorer({ activeDoc }) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await executeSearch(query.trim(), {
        mode,
        topK,
        rerank,
        documentId: activeDoc ? activeDoc.document_id : null,
      });
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="retrieval-card">
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '0.35rem' }}>
          Hybrid Retrieval &amp; Cross-Encoder Reranker Explorer
        </h2>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          Compare Dense Vector semantics with Sparse BM25 lexical matching fused with Reciprocal Rank Fusion (RRF).
        </p>
      </div>

      <form onSubmit={handleSearch}>
        <div className="search-bar-row">
          <input
            type="text"
            className="chat-input-text"
            placeholder="Search passages (e.g. 'Dynamic dispatch vtable resolution runtime polymorphism')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" className="btn-primary" disabled={isLoading || !query.trim()}>
            {isLoading ? 'Retrieving...' : 'Search'}
          </button>
        </div>

        <div className="search-controls-strip">
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600 }}>
            <span>Retrieval Mode:</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              style={{
                fontFamily: 'var(--font-sans)',
                padding: '0.25rem 0.5rem',
                borderRadius: '6px',
                border: '1px solid var(--border-light)',
                background: '#ffffff',
              }}
            >
              <option value="hybrid">Hybrid (Dense Vector + BM25 Sparse)</option>
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
              onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              style={{
                width: '55px',
                padding: '0.25rem 0.5rem',
                borderRadius: '6px',
                border: '1px solid var(--border-light)',
                background: '#ffffff',
              }}
            />
          </label>
        </div>
      </form>

      {error && (
        <div style={{
          padding: '0.85rem',
          borderRadius: '8px',
          background: 'var(--rose-50)',
          color: 'var(--rose-600)',
          fontSize: '0.85rem',
          marginBottom: '1rem',
        }}>
          Search error: {error}
        </div>
      )}

      {/* Results Feed */}
      <div className="search-hits-grid">
        {results ? (
          <>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              marginBottom: '0.5rem',
            }}>
              <span>
                Found <strong>{results.total_hits}</strong> hits in <strong>{results.execution_time_ms} ms</strong>
              </span>
              <span>Retriever: <strong>{results.mode}</strong></span>
            </div>

            {results.hits.map((hit, idx) => (
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
          <div style={{ textAlign: 'center', padding: '3.5rem', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Submit a query above to inspect raw candidate passages from the active document.
          </div>
        )}
      </div>
    </div>
  );
}

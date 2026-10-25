import React from 'react';

export default function DocumentSelector({ documents, selectedDocId, onSelectDoc }) {
  const activeDoc = documents.find(d => d.document_id === selectedDocId);
  const totalChunks = documents.reduce((acc, d) => acc + (d.num_chunks || 0), 0);
  const totalPages = documents.reduce((acc, d) => acc + (d.num_pages || 0), 0);

  return (
    <div className="document-context-bar">
      <div className="doc-selector-group">
        <span className="doc-selector-label">Target Corpus:</span>
        <select
          className="doc-dropdown"
          value={selectedDocId}
          onChange={(e) => onSelectDoc(e.target.value)}
        >
          <option value="all">
            🌐 All Indexed Documents ({documents.length} files • {totalChunks} chunks)
          </option>
          {documents.map((doc) => (
            <option key={doc.document_id} value={doc.document_id}>
              📄 {doc.filename} ({doc.num_pages} pages • {doc.num_chunks} chunks)
            </option>
          ))}
        </select>
      </div>

      <div className="doc-stats-summary">
        {activeDoc ? (
          <>
            <span className="doc-stat-item">
              Document: <strong>{activeDoc.filename}</strong>
            </span>
            <span className="doc-stat-item">
              Pages: <strong>{activeDoc.num_pages}</strong>
            </span>
            <span className="doc-stat-item">
              Vector Chunks: <strong>{activeDoc.num_chunks}</strong>
            </span>
            <span className="doc-stat-item">
              Storage: <strong>{(activeDoc.size_bytes / 1024).toFixed(1)} KB</strong>
            </span>
            <span className="doc-stat-item">
              Status: <strong style={{ color: '#059669' }}>Indexed &amp; Ready</strong>
            </span>
          </>
        ) : (
          <span className="doc-stat-item">
            Query Scope: <strong>Entire Corpus ({documents.length} Documents, {totalPages} Pages, {totalChunks} Chunks)</strong>
          </span>
        )}
      </div>
    </div>
  );
}

import React from 'react';

export default function CitationModal({ citation, onClose }) {
  if (!citation) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 800 }}>Source Context Passage</h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {citation.filename || 'Document'} • Page {citation.page_number || 1} • Chunk #{citation.chunk_id || 'N/A'}
            </p>
          </div>
          <button className="btn-icon" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div style={{
            background: 'var(--bg-subtle)',
            padding: '1rem',
            borderRadius: '8px',
            border: '1px solid var(--border-light)',
            fontSize: '0.85rem',
            lineHeight: 1.6,
            color: 'var(--text-primary)',
            maxHeight: '350px',
            overflowY: 'auto',
          }}>
            {citation.text}
          </div>
        </div>
      </div>
    </div>
  );
}

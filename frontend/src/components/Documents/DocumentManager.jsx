import React, { useState } from 'react';

export default function DocumentManager({
  documents,
  isUploading,
  uploadProgressMsg,
  onUploadFile,
  onSelectAndChat,
}) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div>
      {/* Upload Drop Zone */}
      <div
        className="upload-drop-zone"
        style={{
          borderColor: isDragOver ? 'var(--indigo-600)' : 'var(--border-strong)',
          background: isDragOver ? 'var(--indigo-50)' : '#ffffff',
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-upload-input').click()}
      >
        <input
          id="file-upload-input"
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files.length > 0) {
              onUploadFile(e.target.files[0]);
            }
          }}
        />

        <div className="drop-icon-box">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>

        <h3 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '0.35rem' }}>
          Click or Drag &amp; Drop Any PDF Document Here
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '540px', margin: '0 auto' }}>
          RAGForge automatically extracts PDF pages, runs hierarchical parent-child chunking, generates dense embeddings, extracts tables &amp; figures, and registers entities in the Knowledge Graph.
        </p>

        {isUploading && (
          <div style={{
            marginTop: '1.25rem',
            padding: '0.75rem 1.25rem',
            borderRadius: '8px',
            background: 'var(--indigo-50)',
            color: 'var(--indigo-600)',
            fontWeight: 700,
            fontSize: '0.88rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            <div style={{
              width: '16px',
              height: '16px',
              border: '2px solid var(--indigo-600)',
              borderTopColor: 'transparent',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}></div>
            <span>{uploadProgressMsg || 'Ingesting and vectorizing document...'}</span>
          </div>
        )}
      </div>

      {/* Uploaded Documents Table */}
      <div className="docs-table-wrapper">
        <table className="modern-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Document ID</th>
              <th>Size</th>
              <th>Pages</th>
              <th>Vector Chunks</th>
              <th>Extracted Images</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.document_id}>
                <td>
                  <strong>{doc.filename}</strong>
                </td>
                <td>
                  <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                    {doc.document_id}
                  </code>
                </td>
                <td>{(doc.size_bytes / 1024).toFixed(1)} KB</td>
                <td>{doc.num_pages}</td>
                <td>
                  <span style={{ fontWeight: 700, color: 'var(--indigo-600)' }}>
                    {doc.num_chunks}
                  </span>
                </td>
                <td>
                  <span>{doc.num_images || 0}</span>
                </td>
                <td>
                  <button
                    className="btn-primary"
                    style={{
                      padding: '0.3rem 0.75rem',
                      fontSize: '0.78rem',
                    }}
                    onClick={() => onSelectAndChat(doc.document_id)}
                  >
                    Select &amp; Chat
                  </button>
                </td>
              </tr>
            ))}

            {documents.length === 0 && (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)' }}>
                  No documents found. Drop a PDF above to begin.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

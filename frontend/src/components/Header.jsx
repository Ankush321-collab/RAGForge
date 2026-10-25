import React from 'react';

export default function Header({ services, onRefreshHealth }) {
  return (
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
          <p>Enterprise Multimodal Agentic RAG Platform</p>
        </div>
      </div>

      <div className="infra-strip">
        <div className="status-pill" title="FastAPI API Gateway">
          <span className={`status-dot ${services.backend}`}></span>
          <span>FastAPI :8000</span>
        </div>
        <div className="status-pill" title="MinIO S3 Object Storage">
          <span className={`status-dot ${services.minio}`}></span>
          <span>MinIO S3 :9000</span>
        </div>
        <div className="status-pill" title="OpenSearch Vector & BM25 Store">
          <span className={`status-dot ${services.opensearch}`}></span>
          <span>OpenSearch :9200</span>
        </div>
        <div className="status-pill" title="Neo4j Knowledge Graph">
          <span className={`status-dot ${services.neo4j}`}></span>
          <span>Neo4j :7474</span>
        </div>
        <button className="btn-icon" onClick={onRefreshHealth} title="Refresh Service Status">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
          </svg>
        </button>
      </div>
    </header>
  );
}

/**
 * RAGForge API Client
 * Centralized service layer for communicating with the FastAPI backend.
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : '';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchServiceStatus() {
  const status = {
    backend: 'offline',
    minio: 'offline',
    opensearch: 'offline',
    neo4j: 'offline',
  };

  // Backend
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) status.backend = 'online';
  } catch {
    status.backend = 'offline';
  }

  // MinIO
  try {
    await fetch('http://localhost:9000/minio/health/live', { mode: 'no-cors' });
    status.minio = 'online';
  } catch {
    status.minio = 'offline';
  }

  // OpenSearch
  try {
    await fetch('http://localhost:9200', { mode: 'no-cors' });
    status.opensearch = 'online';
  } catch {
    status.opensearch = 'offline';
  }

  // Neo4j
  try {
    await fetch('http://localhost:7474', { mode: 'no-cors' });
    status.neo4j = 'online';
  } catch {
    status.neo4j = 'offline';
  }

  return status;
}

export async function fetchDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.statusText}`);
  return res.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/documents`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `Upload failed: ${res.statusText}`);
  }

  return res.json();
}

export async function fetchDocumentGraph(documentId) {
  const res = await fetch(`${API_BASE}/documents/${documentId}/graph`);
  if (!res.ok) throw new Error(`Failed to fetch document graph: ${res.statusText}`);
  return res.json();
}

export async function fetchDocumentChunks(documentId, limit = 10) {
  const res = await fetch(`${API_BASE}/documents/${documentId}/chunks?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch document chunks: ${res.statusText}`);
  return res.json();
}

export async function sendChatMessage(question, documentId = null, k = 5, chatHistory = []) {
  const payload = {
    question,
    k,
    chat_history: chatHistory,
    document_id: documentId && documentId !== 'all' ? documentId : null,
  };

  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `Chat request failed: ${res.statusText}`);
  }

  return res.json();
}

export async function executeSearch(query, options = {}) {
  const {
    mode = 'hybrid',
    topK = 5,
    rerank = true,
    documentId = null,
  } = options;

  const payload = {
    query,
    mode,
    top_k: topK,
    rerank,
    document_id: documentId && documentId !== 'all' ? documentId : null,
  };

  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `Search execution failed: ${res.statusText}`);
  }

  return res.json();
}

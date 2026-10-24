const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? 'http://localhost:8000' 
  : '';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initStatusChecker();
  initChat();
  initSearch();
  initDocumentUpload();
});

// TAB SWITCHING
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add('active');
    });
  });
}

// STATUS CHECKER
async function checkService(url, badgeId) {
  const badge = document.getElementById(badgeId);
  if (!badge) return;
  try {
    const res = await fetch(url, { method: 'GET', mode: 'cors' });
    if (res.ok || res.status === 401 || res.status === 403 || res.status === 200) {
      badge.classList.remove('offline');
      badge.classList.add('online');
      return true;
    }
  } catch (err) {
    // If direct fetch fails (e.g. CORS), try via backend proxy or mark offline
  }
  badge.classList.remove('online');
  badge.classList.add('offline');
  return false;
}

async function refreshHealthStatus() {
  const healthOutput = document.getElementById('health-json-output');

  // Check Backend
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById('badge-backend')?.classList.add('online');
      document.getElementById('badge-backend')?.classList.remove('offline');
      if (healthOutput) healthOutput.textContent = JSON.stringify(data, null, 2);
    } else {
      document.getElementById('badge-backend')?.classList.add('offline');
    }
  } catch (err) {
    document.getElementById('badge-backend')?.classList.add('offline');
    if (healthOutput) healthOutput.textContent = `Backend not reachable at ${API_BASE}: ${err.message}`;
  }

  // Check MinIO API (9000/minio/health/live)
  try {
    const res = await fetch('http://localhost:9000/minio/health/live', { mode: 'no-cors' });
    document.getElementById('badge-minio')?.classList.add('online');
    document.getElementById('badge-minio')?.classList.remove('offline');
  } catch {
    document.getElementById('badge-minio')?.classList.add('offline');
  }

  // Check OpenSearch (9200)
  try {
    const res = await fetch('http://localhost:9200', { mode: 'no-cors' });
    document.getElementById('badge-opensearch')?.classList.add('online');
    document.getElementById('badge-opensearch')?.classList.remove('offline');
  } catch {
    document.getElementById('badge-opensearch')?.classList.add('offline');
  }

  // Check Neo4j (7474)
  try {
    const res = await fetch('http://localhost:7474', { mode: 'no-cors' });
    document.getElementById('badge-neo4j')?.classList.add('online');
    document.getElementById('badge-neo4j')?.classList.remove('offline');
  } catch {
    document.getElementById('badge-neo4j')?.classList.add('offline');
  }
}

function initStatusChecker() {
  const btn = document.getElementById('btn-refresh-status');
  if (btn) btn.addEventListener('click', refreshHealthStatus);
  refreshHealthStatus();
}

// AGENTIC CHAT
function initChat() {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const messagesContainer = document.getElementById('chat-messages');
  const chips = document.querySelectorAll('.query-chip');

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      if (input) {
        input.value = chip.getAttribute('data-query');
        form.dispatchEvent(new Event('submit'));
      }
    });
  });

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Append User Message
    appendMessage('user', query);
    input.value = '';

    // Append Loading Assistant Message
    const loadingId = 'loading-' + Date.now();
    appendMessage('assistant', '<span class="typing-indicator">Retrieving across local vectors, BM25 & graph...</span>', loadingId);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, k: 5 })
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
      }

      const data = await res.json();
      removeMessage(loadingId);
      appendMessage('assistant', formatAnswer(data.answer));

      // Update Inspector
      updateInspector(data);
    } catch (err) {
      removeMessage(loadingId);
      appendMessage('assistant', `<span style="color: var(--error)">Error querying backend: ${err.message}</span>`);
    }
  });
}

function appendMessage(sender, textHtml, id = null) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  const msg = document.createElement('div');
  msg.className = `chat-message ${sender}`;
  if (id) msg.id = id;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = sender === 'user' ? 'YOU' : 'AI';

  const content = document.createElement('div');
  content.className = 'message-content';
  content.innerHTML = textHtml;

  msg.appendChild(avatar);
  msg.appendChild(content);
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function formatAnswer(text) {
  if (!text) return 'No response generated.';
  return text
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function updateInspector(data) {
  const valRoute = document.getElementById('val-route');
  const valGroundingBar = document.getElementById('val-grounding-bar');
  const valGroundingNum = document.getElementById('val-grounding-num');
  const valGroundingStatus = document.getElementById('val-grounding-status');
  const citationsList = document.getElementById('citations-list');

  if (valRoute) valRoute.textContent = data.route || 'Passage Retrieval';

  const score = Math.round((data.grounding_score || 0) * 100);
  if (valGroundingBar) valGroundingBar.style.width = `${score}%`;
  if (valGroundingNum) valGroundingNum.textContent = `${score}% (${data.grounding_score?.toFixed(2) || '0.00'})`;
  if (valGroundingStatus) {
    valGroundingStatus.textContent = data.is_valid ? 'Validated & Grounded' : 'Low Grounding Confidence';
    valGroundingStatus.style.color = data.is_valid ? 'var(--success)' : 'var(--warning)';
  }

  if (citationsList && data.citations) {
    if (data.citations.length === 0) {
      citationsList.innerHTML = '<p class="placeholder-text">No direct passage citations found.</p>';
    } else {
      citationsList.innerHTML = data.citations.map((c, i) => `
        <div class="citation-card">
          <strong>#${i + 1} Document:</strong> ${c.document_id || 'oops_notes'}<br>
          <strong>Page:</strong> ${c.page_number || 1} | <strong>Chunk:</strong> ${c.chunk_id || c.child_id || 'c_' + i}
        </div>
      `).join('');
    }
  }
}

// HYBRID SEARCH
function initSearch() {
  const form = document.getElementById('search-form');
  const input = document.getElementById('search-query-input');
  const modeSelect = document.getElementById('search-mode-select');
  const topKInput = document.getElementById('search-top-k');
  const rerankCheck = document.getElementById('search-rerank-check');
  const resultsMeta = document.getElementById('search-results-meta');
  const hitsContainer = document.getElementById('search-hits-container');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    hitsContainer.innerHTML = '<div class="empty-state"><p>Searching across vectors and inverted index...</p></div>';

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          mode: modeSelect.value,
          top_k: parseInt(topKInput.value) || 5,
          rerank: rerankCheck.checked
        })
      });

      if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
      const data = await res.json();

      // Update meta
      if (resultsMeta) {
        resultsMeta.style.display = 'flex';
        document.getElementById('meta-hit-count').textContent = data.total_hits;
        document.getElementById('meta-exec-time').textContent = `${data.execution_time_ms} ms`;
        document.getElementById('meta-mode').textContent = data.mode;
      }

      if (!data.hits || data.hits.length === 0) {
        hitsContainer.innerHTML = '<div class="empty-state"><p>No relevant document passages matched your query.</p></div>';
        return;
      }

      hitsContainer.innerHTML = data.hits.map((h, i) => `
        <div class="search-hit-card">
          <div class="hit-meta">
            <span><strong>Chunk ID:</strong> ${h.chunk_id || h.id || i} (Page ${h.page_number || 1})</span>
            <div class="hit-scores">
              ${h.score !== null ? `<span class="score-badge" title="RRF / Retrieval Score">Score: ${Number(h.score).toFixed(4)}</span>` : ''}
              ${h.rerank_score !== null && h.rerank_score !== undefined ? `<span class="score-badge" title="Cross-Encoder Rerank Score" style="background: rgba(16, 185, 129, 0.15); color: #6ee7b7">Rerank: ${Number(h.rerank_score).toFixed(4)}</span>` : ''}
            </div>
          </div>
          <p class="hit-text">${escapeHtml(h.text)}</p>
        </div>
      `).join('');
    } catch (err) {
      hitsContainer.innerHTML = `<div class="empty-state"><p style="color: var(--error)">Search error: ${err.message}</p></div>`;
    }
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// DOCUMENT UPLOAD
function initDocumentUpload() {
  const fileInput = document.getElementById('file-input');
  const dropzoneBox = document.getElementById('dropzone-box');
  const selectedLabel = document.getElementById('selected-filename');
  const btnUpload = document.getElementById('btn-upload');
  const uploadForm = document.getElementById('upload-form');
  const alertBox = document.getElementById('upload-status-alert');
  const tableBody = document.getElementById('documents-table-body');

  if (!dropzoneBox || !fileInput) return;

  dropzoneBox.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      selectedLabel.textContent = `Selected: ${fileInput.files[0].name} (${(fileInput.files[0].size / 1024).toFixed(1)} KB)`;
      btnUpload.disabled = false;
    }
  });

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    btnUpload.disabled = true;
    btnUpload.textContent = 'Uploading to MinIO S3...';
    alertBox.style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const res = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errDetail = await res.text();
        throw new Error(errDetail || res.statusText);
      }

      const doc = await res.json();
      alertBox.className = 'alert-box success';
      alertBox.textContent = `Document successfully uploaded to local S3! ID: ${doc.document_id}`;
      alertBox.style.display = 'block';

      // Add to table
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${escapeHtml(doc.filename)}</strong></td>
        <td><code>${doc.document_id}</code></td>
        <td>${(doc.size_bytes / 1024).toFixed(1)} KB</td>
        <td><span class="badge-s3">MinIO / Local</span></td>
        <td>
          <a href="${API_BASE}/documents/download/${doc.object_key}" target="_blank" class="btn-sm">Download</a>
        </td>
      `;
      tableBody.prepend(tr);

      // Reset
      fileInput.value = '';
      selectedLabel.textContent = '';
      btnUpload.textContent = 'Upload to Local S3';
    } catch (err) {
      alertBox.className = 'alert-box error';
      alertBox.textContent = `Upload failed: ${err.message}`;
      alertBox.style.display = 'block';
      btnUpload.disabled = false;
      btnUpload.textContent = 'Upload to Local S3';
    }
  });
}

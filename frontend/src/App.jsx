import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import DocumentSelector from './components/DocumentSelector';
import Navigation from './components/Navigation';
import ChatView from './components/Chat/ChatView';
import RetrievalExplorer from './components/Retrieval/RetrievalExplorer';
import DocumentManager from './components/Documents/DocumentManager';
import KnowledgeGraphView from './components/Graph/KnowledgeGraphView';
import {
  fetchDocuments,
  fetchServiceStatus,
  fetchDocumentGraph,
  uploadDocument,
  sendChatMessage,
} from './api/client';

export default function App() {
  // Navigation & Documents State
  const [activeTab, setActiveTab] = useState('chat');
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState('all');

  // Service Health
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
      content: 'Welcome to RAGForge! Upload any PDF document or choose an existing document to ask questions with real-time citation verification and grounding checks.',
      citations: [],
      groundingScore: 1.0,
      isValid: true,
    },
  ]);
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Upload State
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressMsg, setUploadProgressMsg] = useState('');

  // Graph State
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isGraphLoading, setIsGraphLoading] = useState(false);

  // Initialize
  useEffect(() => {
    loadDocuments();
    refreshHealth();
  }, []);

  // Sync Knowledge Graph with active document
  useEffect(() => {
    if (selectedDocId && selectedDocId !== 'all') {
      loadGraph(selectedDocId);
    }
  }, [selectedDocId]);

  const loadDocuments = async () => {
    try {
      const data = await fetchDocuments();
      setDocuments(data);
    } catch (err) {
      console.warn("Could not load documents:", err);
    }
  };

  const refreshHealth = async () => {
    try {
      const status = await fetchServiceStatus();
      setServices(status);
    } catch {
      // Ignored
    }
  };

  const loadGraph = async (docId) => {
    setIsGraphLoading(true);
    try {
      const data = await fetchDocumentGraph(docId);
      setGraphData(data);
    } catch (err) {
      console.warn("Graph load error:", err);
    } finally {
      setIsGraphLoading(false);
    }
  };

  // Upload Any PDF with Full Pipeline
  const handleUploadFile = async (file) => {
    if (!file || !file.name.endsWith('.pdf')) {
      alert("Please upload a valid PDF document (.pdf)");
      return;
    }

    setIsUploading(true);
    setUploadProgressMsg(`Ingesting & chunking "${file.name}"...`);

    try {
      const newDoc = await uploadDocument(file);
      setUploadProgressMsg(`Indexed ${newDoc.num_pages} pages into ${newDoc.num_chunks} vector chunks!`);

      // Refresh list & select document
      await loadDocuments();
      setSelectedDocId(newDoc.document_id);

      // Append assistant confirmation
      setChatMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          route: 'INGESTION_COMPLETE',
          content: `Document **"${newDoc.filename}"** is now fully indexed!\n\n• **Pages Extracted:** ${newDoc.num_pages}\n• **Parent-Child Chunks:** ${newDoc.num_chunks}\n• **Vector Index:** Active in ChromaDB\n\nYou can now ask any question specifically about this document.`,
          citations: [],
          groundingScore: 1.0,
          isValid: true,
        },
      ]);

      setTimeout(() => {
        setIsUploading(false);
        setUploadProgressMsg('');
        setActiveTab('chat');
      }, 1200);
    } catch (err) {
      alert(`Upload error: ${err.message}`);
      setIsUploading(false);
      setUploadProgressMsg('');
    }
  };

  // Send Chat Message
  const handleSendMessage = async (query) => {
    // Append user message
    setChatMessages((prev) => [...prev, { role: 'user', content: query }]);
    setIsChatLoading(true);

    try {
      const history = chatMessages.slice(-4).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await sendChatMessage(query, selectedDocId, 5, history);

      setChatMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          route: response.route,
          content: response.answer,
          citations: response.citations || [],
          groundingScore: response.grounding_score,
          isValid: response.is_valid,
          context: response.context,
        },
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          route: 'ERROR',
          content: `Failed to retrieve answer: ${err.message}. Please check if the FastAPI backend is running.`,
          citations: [],
          groundingScore: 0.0,
          isValid: false,
        },
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const activeDoc = documents.find((d) => d.document_id === selectedDocId);

  return (
    <div className="app-shell">
      {/* 1. Header with Microservice Health */}
      <Header
        services={services}
        onRefreshHealth={refreshHealth}
      />

      {/* 2. Active Document Selector Bar */}
      <DocumentSelector
        documents={documents}
        selectedDocId={selectedDocId}
        onSelectDoc={setSelectedDocId}
      />

      {/* 3. Navigation Bar */}
      <Navigation
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* 4. Main Views */}
      <main className="dashboard-main">
        {activeTab === 'chat' && (
          <ChatView
            messages={chatMessages}
            isLoading={isChatLoading}
            onSendMessage={handleSendMessage}
            activeDoc={activeDoc}
          />
        )}

        {activeTab === 'search' && (
          <RetrievalExplorer
            activeDoc={activeDoc}
          />
        )}

        {activeTab === 'documents' && (
          <DocumentManager
            documents={documents}
            isUploading={isUploading}
            uploadProgressMsg={uploadProgressMsg}
            onUploadFile={handleUploadFile}
            onSelectAndChat={(docId) => {
              setSelectedDocId(docId);
              setActiveTab('chat');
            }}
          />
        )}

        {activeTab === 'graph' && (
          <KnowledgeGraphView
            graphData={graphData}
            isLoading={isGraphLoading}
            activeDoc={activeDoc}
            onRefreshGraph={() => {
              if (selectedDocId && selectedDocId !== 'all') {
                loadGraph(selectedDocId);
              }
            }}
          />
        )}
      </main>
    </div>
  );
}

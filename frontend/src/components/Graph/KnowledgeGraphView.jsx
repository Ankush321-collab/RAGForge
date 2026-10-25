import React, { useState } from 'react';

export default function KnowledgeGraphView({
  graphData,
  isLoading,
  activeDoc,
  onRefreshGraph,
}) {
  const [selectedNode, setSelectedNode] = useState(null);

  const nodes = graphData.nodes || [];
  const links = graphData.links || [];

  return (
    <div className="graph-stage-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 800 }}>
            Neo4j Knowledge Graph Multi-Hop Traversal
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Entities and directed relationship edges extracted from <strong>{activeDoc ? activeDoc.filename : 'the active corpus'}</strong>.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="status-pill">
            Entities: <strong>{nodes.length}</strong>
          </span>
          <span className="status-pill">
            Relationships: <strong>{links.length}</strong>
          </span>
          <button
            className="btn-primary"
            style={{ padding: '0.35rem 0.8rem', fontSize: '0.78rem' }}
            onClick={onRefreshGraph}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Refresh Graph'}
          </button>
        </div>
      </div>

      {/* Interactive SVG Canvas */}
      <div className="graph-canvas-frame">
        {isLoading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
            Querying Neo4j knowledge graph traversal...
          </div>
        ) : nodes.length > 0 ? (
          <svg width="100%" height="100%" viewBox="0 0 860 460" style={{ overflow: 'visible' }}>
            <defs>
              <marker
                id="graph-arrowhead"
                viewBox="0 0 10 10"
                refX="25"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
              </marker>
            </defs>

            {/* Directed Relationship Edges */}
            {links.slice(0, 20).map((link, idx) => {
              const srcIdx = nodes.findIndex((n) => n.id === link.source);
              const tgtIdx = nodes.findIndex((n) => n.id === link.target);

              const sIndex = srcIdx >= 0 ? srcIdx : idx % 5;
              const tIndex = tgtIdx >= 0 ? tgtIdx : (idx + 1) % 5;

              const x1 = 120 + (sIndex % 4) * 200;
              const y1 = 90 + Math.floor(sIndex / 4) * 120;
              const x2 = 120 + (tIndex % 4) * 200;
              const y2 = 90 + Math.floor(tIndex / 4) * 120;

              return (
                <g key={idx}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#cbd5e1"
                    strokeWidth="2"
                    markerEnd="url(#graph-arrowhead)"
                  />
                  <rect
                    x={(x1 + x2) / 2 - 28}
                    y={(y1 + y2) / 2 - 8}
                    width="56"
                    height="16"
                    rx="4"
                    fill="#ffffff"
                    stroke="#e2e8f0"
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 + 3}
                    fontSize="8"
                    fontWeight="700"
                    fill="#64748b"
                    textAnchor="middle"
                  >
                    {link.label}
                  </text>
                </g>
              );
            })}

            {/* Entity Nodes */}
            {nodes.slice(0, 16).map((node, idx) => {
              const cx = 120 + (idx % 4) * 200;
              const cy = 90 + Math.floor(idx / 4) * 120;
              const isSelected = selectedNode?.id === node.id;
              const isConcept = node.type === 'Concept';

              return (
                <g
                  key={idx}
                  onClick={() => setSelectedNode(node)}
                  style={{ cursor: 'pointer', transition: 'transform 0.15s ease' }}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 36 : 32}
                    fill={isConcept ? '#eef2ff' : '#ecfdf5'}
                    stroke={isSelected ? '#4f46e5' : isConcept ? '#6366f1' : '#059669'}
                    strokeWidth={isSelected ? 3.5 : 2}
                  />
                  <text
                    x={cx}
                    y={cy + 4}
                    fontSize="9"
                    fontWeight="800"
                    fill="#0f172a"
                    textAnchor="middle"
                  >
                    {node.label.slice(0, 15)}
                  </text>
                </g>
              );
            })}
          </svg>
        ) : (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--text-muted)',
            gap: '0.75rem',
          }}>
            <p>No Knowledge Graph entities extracted yet for this document.</p>
            <button className="btn-primary" onClick={onRefreshGraph}>
              Extract Graph Entities Now
            </button>
          </div>
        )}
      </div>

      {/* Selected Node Details or Cypher Inspector */}
      <div style={{
        background: 'var(--bg-subtle)',
        border: '1px solid var(--border-light)',
        borderRadius: '8px',
        padding: '0.85rem 1rem',
        fontSize: '0.82rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        {selectedNode ? (
          <div>
            <strong>Selected Entity:</strong> {selectedNode.label} ({selectedNode.type})
          </div>
        ) : (
          <span style={{ color: 'var(--text-muted)' }}>
            Click on any graph entity above to inspect relations and multi-hop Cypher paths.
          </span>
        )}

        <code style={{
          fontFamily: 'var(--font-mono)',
          background: '#ffffff',
          padding: '0.2rem 0.5rem',
          borderRadius: '4px',
          border: '1px solid var(--border-light)',
          fontSize: '0.75rem',
          color: 'var(--indigo-600)',
        }}>
          MATCH (e)-[r]-&gt;(t) RETURN e, r, t LIMIT 25
        </code>
      </div>
    </div>
  );
}

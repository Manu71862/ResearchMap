import React, { useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  MarkerType,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import {
  GitFork,
  FileText,
  Quote,
  CheckCircle,
  XCircle,
  AlertCircle,
  Maximize2
} from 'lucide-react';

export default function EvidenceGraph({ result, onSelectNode }) {
  if (!result) return null;

  const {
    claim,
    retrieved_papers = [],
    evidence_details = [],
    predicted_label
  } = result;

  // Build nodes and edges dynamically from real backend verification results
  const { initialNodes, initialEdges } = useMemo(() => {
    const nodes = [];
    const edges = [];

    const claimText = claim.length > 90 ? claim.slice(0, 87) + '...' : claim;

    // 1. Root Claim Node
    const claimNodeId = 'claim-root';
    nodes.push({
      id: claimNodeId,
      type: 'default',
      position: { x: 420, y: 30 },
      data: {
        nodeType: 'claim',
        title: 'Scientific Claim',
        text: claim,
        predicted_label,
        label: (
          <div style={{ textAlign: 'left', maxWidth: '280px', padding: '4px' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.7rem',
              color: 'var(--accent-cyan)',
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              fontWeight: 600,
              marginBottom: '4px'
            }}>
              <GitFork size={12} /> Claim Under Test
            </div>
            <div style={{ fontSize: '0.8rem', color: '#f8fafc', lineHeight: '1.35', fontWeight: 500 }}>
              {claimText}
            </div>
          </div>
        )
      },
      style: {
        background: '#0f172a',
        border: '2px solid var(--accent-cyan)',
        borderRadius: '10px',
        boxShadow: '0 0 20px rgba(56, 189, 248, 0.25)',
        width: 300,
      }
    });

    // 2. Paper Nodes
    const paperSpacingX = 320;
    const startX = Math.max(40, 420 - ((retrieved_papers.length - 1) * paperSpacingX) / 2);

    retrieved_papers.forEach((paper, pIdx) => {
      const paperNodeId = `paper-${paper.doc_id}`;
      const px = startX + pIdx * paperSpacingX;
      const py = 200;

      const paperTitle = paper.title.length > 55 ? paper.title.slice(0, 52) + '...' : paper.title;

      nodes.push({
        id: paperNodeId,
        type: 'default',
        position: { x: px, y: py },
        data: {
          nodeType: 'paper',
          doc_id: paper.doc_id,
          title: paper.title,
          bm25_score: paper.bm25_score,
          abstract_preview: paper.abstract_preview,
          label: (
            <div style={{ textAlign: 'left', maxWidth: '240px', padding: '4px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.68rem',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                marginBottom: '4px'
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#93c5fd' }}>
                  <FileText size={11} /> Doc #{paper.doc_id}
                </span>
                <span style={{ background: '#1e293b', padding: '1px 5px', borderRadius: '4px' }}>
                  BM25: {paper.bm25_score.toFixed(1)}
                </span>
              </div>
              <div style={{ fontSize: '0.78rem', color: '#e2e8f0', lineHeight: '1.3', fontWeight: 500 }}>
                {paperTitle}
              </div>
            </div>
          )
        },
        style: {
          background: '#131c31',
          border: '1px solid #334155',
          borderRadius: '8px',
          width: 260,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        }
      });

      // Edge from Claim to Paper
      edges.push({
        id: `edge-${claimNodeId}-${paperNodeId}`,
        source: claimNodeId,
        target: paperNodeId,
        animated: true,
        style: { stroke: '#38bdf8', strokeWidth: 1.5, strokeDasharray: '4 4' },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#38bdf8', width: 14, height: 14 },
      });

      // 3. Evidence Sentences belonging to this paper
      const paperEvidence = evidence_details.filter(e => e.doc_id === paper.doc_id);
      paperEvidence.forEach((ev, eIdx) => {
        const evNodeId = `ev-${ev.doc_id}-${ev.sentence_idx}`;
        const ex = px - 30 + eIdx * 140;
        const ey = 380 + (eIdx % 2) * 50; // stagger vertically if multiple

        const isSupp = ev.entailment_label === 'SUPPORTS';
        const isContra = ev.entailment_label === 'CONTRADICTS';

        const borderColor = isSupp ? '#10b981' : isContra ? '#f43f5e' : '#f59e0b';
        const bgColor = isSupp ? 'rgba(16, 185, 129, 0.12)' : isContra ? 'rgba(244, 63, 94, 0.12)' : 'rgba(245, 158, 11, 0.12)';
        const textColor = isSupp ? '#34d399' : isContra ? '#fb7185' : '#fbbf24';

        const snippet = ev.text.length > 50 ? ev.text.slice(0, 47) + '...' : ev.text;

        nodes.push({
          id: evNodeId,
          type: 'default',
          position: { x: ex, y: ey },
          data: {
            nodeType: 'evidence',
            ...ev,
            label: (
              <div style={{ textAlign: 'left', maxWidth: '200px', padding: '2px' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.65rem',
                  color: textColor,
                  fontFamily: 'var(--font-mono)',
                  marginBottom: '2px',
                  fontWeight: 600
                }}>
                  <span>S[{ev.sentence_idx}] {ev.entailment_label}</span>
                  <span>{Math.round(ev.confidence * 100)}%</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#cbd5e1', lineHeight: '1.25' }}>
                  "{snippet}"
                </div>
              </div>
            )
          },
          style: {
            background: bgColor,
            border: `1.5px solid ${borderColor}`,
            borderRadius: '6px',
            width: 220,
            boxShadow: `0 2px 10px ${borderColor}33`
          }
        });

        // Edge from Paper to Evidence
        edges.push({
          id: `edge-${paperNodeId}-${evNodeId}`,
          source: paperNodeId,
          target: evNodeId,
          style: { stroke: borderColor, strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: borderColor, width: 12, height: 12 },
        });
      });
    });

    return { initialNodes: nodes, initialEdges: edges };
  }, [claim, retrieved_papers, evidence_details, predicted_label]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Re-sync when props change
  React.useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const handleNodeClick = useCallback((event, node) => {
    if (onSelectNode && node.data) {
      onSelectNode(node.data);
    }
  }, [onSelectNode]);

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '14px',
      overflow: 'hidden',
      marginBottom: '2rem'
    }}>
      {/* Header bar */}
      <div style={{
        padding: '1rem 1.5rem',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(15, 23, 42, 0.6)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <GitFork size={18} style={{ color: 'var(--accent-cyan)' }} />
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Interactive Scientific Evidence Graph
          </h3>
          <span style={{
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            background: 'rgba(56, 189, 248, 0.1)',
            color: 'var(--accent-cyan)',
            padding: '2px 8px',
            borderRadius: '12px'
          }}>
            Click any node to inspect evidence
          </span>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#93c5fd' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: '#38bdf8' }}></span>
            <span>Retrieved Paper</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#34d399' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: '#10b981' }}></span>
            <span>Supports Sentence</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#fb7185' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: '#f43f5e' }}></span>
            <span>Contradicts Sentence</span>
          </div>
        </div>
      </div>

      {/* Graph Canvas */}
      <div style={{ width: '100%', height: '520px', position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.3}
          maxZoom={1.5}
          attributionPosition="bottom-left"
        >
          <Background color="#1e293b" gap={20} size={1} />
          <Controls />
          <MiniMap
            nodeStrokeColor="#64748b"
            nodeColor={(n) => {
              if (n.data?.nodeType === 'claim') return '#38bdf8';
              if (n.data?.nodeType === 'paper') return '#3b82f6';
              if (n.data?.entailment_label === 'SUPPORTS') return '#10b981';
              if (n.data?.entailment_label === 'CONTRADICTS') return '#f43f5e';
              return '#f59e0b';
            }}
            style={{
              background: '#090d16',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px'
            }}
          />
        </ReactFlow>
      </div>
    </div>
  );
}

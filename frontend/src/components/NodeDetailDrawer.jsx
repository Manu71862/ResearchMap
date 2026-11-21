import React from 'react';
import {
  X,
  FileText,
  GitFork,
  CheckCircle,
  XCircle,
  HelpCircle,
  ExternalLink,
  BarChart2,
  Layers,
  Search
} from 'lucide-react';

export default function NodeDetailDrawer({ selectedNode, onClose, onViewPaper }) {
  if (!selectedNode) return null;

  const {
    nodeType,
    title,
    text,
    doc_id,
    sentence_idx,
    bm25_score,
    entailment_label,
    confidence,
    probabilities = {},
    abstract_preview,
    predicted_label
  } = selectedNode;

  const isSupp = entailment_label === 'SUPPORTS';
  const isContra = entailment_label === 'CONTRADICTS';

  const badgeColor = isSupp ? '#10b981' : isContra ? '#f43f5e' : '#f59e0b';

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      bottom: 0,
      width: '420px',
      maxWidth: '90vw',
      background: 'rgba(15, 23, 42, 0.95)',
      backdropFilter: 'blur(16px)',
      borderLeft: '1px solid var(--border-subtle)',
      boxShadow: '-10px 0 35px rgba(0, 0, 0, 0.6)',
      zIndex: 100,
      display: 'flex',
      flexDirection: 'column',
      animation: 'slideIn 0.25s ease-out'
    }}>
      {/* Drawer Header */}
      <div style={{
        padding: '1.25rem 1.5rem',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {nodeType === 'claim' && <GitFork size={18} style={{ color: 'var(--accent-cyan)' }} />}
          {nodeType === 'paper' && <FileText size={18} style={{ color: 'var(--accent-blue)' }} />}
          {nodeType === 'evidence' && (
            isSupp ? <CheckCircle size={18} style={{ color: '#10b981' }} />
            : isContra ? <XCircle size={18} style={{ color: '#f43f5e' }} />
            : <HelpCircle size={18} style={{ color: '#f59e0b' }} />
          )}
          <span style={{
            fontSize: '0.85rem',
            fontFamily: 'var(--font-mono)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: 'var(--text-secondary)',
            fontWeight: 600
          }}>
            {nodeType === 'claim' ? 'Claim Node Details' : nodeType === 'paper' ? 'Paper Node Details' : 'Evidence Passage Inspector'}
          </span>
        </div>

        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
        >
          <X size={20} />
        </button>
      </div>

      {/* Drawer Content Body */}
      <div style={{ padding: '1.5rem', overflowY: 'auto', flex: 1 }}>
        {/* Case 1: Claim Node */}
        {nodeType === 'claim' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.4rem', fontFamily: 'var(--font-mono)' }}>
                VERIFIED SCIENTIFIC CLAIM
              </div>
              <p style={{ fontSize: '1rem', color: '#f8fafc', lineHeight: '1.5', fontWeight: 500 }}>
                {text}
              </p>
            </div>

            {predicted_label && (
              <div style={{
                background: 'rgba(56, 189, 248, 0.08)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', marginBottom: '0.25rem' }}>
                  OVERALL PREDICTED VERDICT
                </div>
                <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc' }}>
                  {predicted_label}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Case 2: Paper Node */}
        {nodeType === 'paper' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <div style={{
                display: 'inline-block',
                background: 'rgba(56, 189, 248, 0.1)',
                color: 'var(--accent-cyan)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                padding: '2px 8px',
                borderRadius: '4px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                marginBottom: '0.5rem'
              }}>
                SciFact Doc ID: {doc_id}
              </div>
              <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: '1.4' }}>
                {title}
              </h4>
            </div>

            <div style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '0.85rem 1rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>BM25 Retrieval Score</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                {bm25_score !== undefined ? bm25_score.toFixed(3) : 'N/A'}
              </span>
            </div>

            {abstract_preview && (
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.4rem', fontFamily: 'var(--font-mono)' }}>
                  ABSTRACT PREVIEW
                </div>
                <p style={{
                  fontSize: '0.85rem',
                  color: 'var(--text-secondary)',
                  lineHeight: '1.6',
                  fontStyle: 'italic',
                  background: 'rgba(0, 0, 0, 0.2)',
                  padding: '1rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-subtle)'
                }}>
                  "{abstract_preview}"
                </p>
              </div>
            )}

            {onViewPaper && (
              <button
                onClick={() => onViewPaper(doc_id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))',
                  color: '#fff',
                  border: 'none',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  marginTop: '0.5rem'
                }}
              >
                <ExternalLink size={16} />
                <span>View Full Paper Abstract</span>
              </button>
            )}
          </div>
        )}

        {/* Case 3: Evidence Sentence Node */}
        {nodeType === 'evidence' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Title & Doc info */}
            <div>
              <div style={{
                display: 'flex',
                gap: '0.5rem',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                marginBottom: '0.5rem'
              }}>
                <span style={{
                  background: 'rgba(56, 189, 248, 0.1)',
                  color: 'var(--accent-cyan)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  Doc #{doc_id}
                </span>
                <span style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  Sentence [{sentence_idx}]
                </span>
              </div>
              <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: '1.4' }}>
                {title}
              </h5>
            </div>

            {/* Verbatim Sentence Text */}
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
              border: `1.5px solid ${badgeColor}`,
              borderRadius: '8px',
              padding: '1rem'
            }}>
              <div style={{
                fontSize: '0.7rem',
                fontFamily: 'var(--font-mono)',
                color: badgeColor,
                fontWeight: 700,
                textTransform: 'uppercase',
                marginBottom: '0.5rem'
              }}>
                Verbatim Evidence Passage
              </div>
              <p style={{ fontSize: '0.92rem', color: '#f8fafc', lineHeight: '1.6' }}>
                "{text}"
              </p>
            </div>

            {/* Model Entailment Verdict Card */}
            <div style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '1rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  FINE-TUNED NLI VERDICT
                </span>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  color: badgeColor,
                  background: `${badgeColor}18`,
                  border: `1px solid ${badgeColor}40`,
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  {entailment_label}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Confidence</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {Math.round((confidence || 0) * 100)}%
                </span>
              </div>

              {bm25_score !== undefined && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  <span>Sentence BM25 Score</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    {bm25_score.toFixed(3)}
                  </span>
                </div>
              )}
            </div>

            {/* Probabilities Distribution */}
            {probabilities && (
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: '0.5rem' }}>
                  CLASS LOGITS BREAKDOWN
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {Object.entries(probabilities).map(([lbl, p]) => {
                    const probPct = Math.round(p * 100);
                    const isTarget = lbl === entailment_label;
                    const c = lbl === 'SUPPORTS' ? '#10b981' : lbl === 'CONTRADICTS' ? '#f43f5e' : '#f59e0b';
                    return (
                      <div key={lbl}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                          <span style={{ color: isTarget ? '#f8fafc' : 'var(--text-muted)', fontWeight: isTarget ? 600 : 400 }}>
                            {lbl}
                          </span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: isTarget ? c : 'var(--text-muted)' }}>
                            {probPct}%
                          </span>
                        </div>
                        <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${probPct}%`, height: '100%', background: c }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Link to Full Abstract */}
            {onViewPaper && (
              <button
                onClick={() => onViewPaper(doc_id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  background: 'rgba(56, 189, 248, 0.08)',
                  color: 'var(--accent-cyan)',
                  border: '1px solid rgba(56, 189, 248, 0.3)',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  marginTop: '0.5rem'
                }}
              >
                <ExternalLink size={15} />
                <span>View Full Abstract in Modal</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

import React from 'react';
import {
  FileText,
  ExternalLink,
  Search,
  SlidersHorizontal,
  ChevronRight,
  Info
} from 'lucide-react';

export default function EvidenceList({ result, onSelectEvidence, onViewPaper }) {
  if (!result || !result.retrieved_papers || result.retrieved_papers.length === 0) {
    return (
      <div style={{
        background: 'var(--bg-card)',
        border: '1px dashed var(--border-subtle)',
        borderRadius: '12px',
        padding: '2.5rem',
        textAlign: 'center',
        color: 'var(--text-muted)'
      }}>
        <Search size={32} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>No Literature Evidence Retrieved</h4>
        <p style={{ fontSize: '0.85rem', maxWidth: '500px', margin: '0 auto' }}>
          No relevant papers or passages were retrieved from the scientific corpus for this claim with the current BM25 thresholds.
        </p>
      </div>
    );
  }

  const { retrieved_papers, evidence_details = [] } = result;

  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1.25rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileText size={20} style={{ color: 'var(--accent-cyan)' }} />
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Retrieved Papers & Evidence Passages
          </h3>
          <span style={{
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono)',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            padding: '2px 8px',
            borderRadius: '9999px',
            color: 'var(--text-secondary)'
          }}>
            {retrieved_papers.length} Papers Retrieved
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        {retrieved_papers.map((paper, idx) => {
          // Find evidence sentences belonging to this paper
          const paperSentences = evidence_details.filter(e => e.doc_id === paper.doc_id);

          return (
            <div
              key={paper.doc_id}
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '12px',
                padding: '1.5rem',
                transition: 'border-color 0.2s ease',
              }}
            >
              {/* Paper Header */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '1rem',
                marginBottom: '1rem'
              }}>
                <div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.6rem',
                    marginBottom: '0.4rem',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem'
                  }}>
                    <span style={{
                      background: 'rgba(56, 189, 248, 0.1)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid rgba(56, 189, 248, 0.25)',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontWeight: 600
                    }}>
                      Doc ID: {paper.doc_id}
                    </span>
                    <span style={{
                      background: 'rgba(15, 23, 42, 0.6)',
                      color: 'var(--text-muted)',
                      border: '1px solid var(--border-subtle)',
                      padding: '2px 8px',
                      borderRadius: '4px'
                    }}>
                      Rank #{idx + 1} • BM25 Score: {paper.bm25_score.toFixed(2)}
                    </span>
                  </div>
                  <h4 style={{
                    fontSize: '1.05rem',
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                    lineHeight: '1.4'
                  }}>
                    {paper.title}
                  </h4>
                </div>

                {/* View Abstract Button */}
                <button
                  onClick={() => onViewPaper(paper.doc_id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    background: 'rgba(56, 189, 248, 0.08)',
                    color: 'var(--accent-cyan)',
                    border: '1px solid rgba(56, 189, 248, 0.3)',
                    padding: '0.45rem 0.85rem',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: 500,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(56, 189, 248, 0.18)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(56, 189, 248, 0.08)'}
                >
                  <ExternalLink size={13} />
                  <span>Full Abstract</span>
                </button>
              </div>

              {/* Abstract Preview */}
              <p style={{
                fontSize: '0.875rem',
                color: 'var(--text-secondary)',
                lineHeight: '1.6',
                marginBottom: '1.25rem',
                fontStyle: 'italic',
                borderLeft: '2px solid var(--border-subtle)',
                paddingLeft: '0.75rem'
              }}>
                "{paper.abstract_preview}"
              </p>

              {/* Extracted Evidence Sentences */}
              {paperSentences.length > 0 && (
                <div style={{
                  background: 'rgba(15, 23, 42, 0.5)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  padding: '1rem'
                }}>
                  <div style={{
                    fontSize: '0.75rem',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    marginBottom: '0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem'
                  }}>
                    <SlidersHorizontal size={13} /> Extracted Verbatim Evidence Sentences ({paperSentences.length})
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {paperSentences.map((sent) => {
                      const isSupp = sent.entailment_label === 'SUPPORTS';
                      const isContra = sent.entailment_label === 'CONTRADICTS';

                      const badgeColor = isSupp ? '#10b981' : isContra ? '#f43f5e' : '#f59e0b';
                      const badgeBg = isSupp ? 'rgba(16, 185, 129, 0.12)' : isContra ? 'rgba(244, 63, 94, 0.12)' : 'rgba(245, 158, 11, 0.12)';
                      const badgeBorder = isSupp ? 'rgba(16, 185, 129, 0.3)' : isContra ? 'rgba(244, 63, 94, 0.3)' : 'rgba(245, 158, 11, 0.3)';

                      return (
                        <div
                          key={sent.sentence_idx}
                          style={{
                            background: badgeBg,
                            border: `1px solid ${badgeBorder}`,
                            borderRadius: '8px',
                            padding: '0.85rem 1rem',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.5rem',
                            cursor: 'pointer',
                            transition: 'transform 0.15s ease'
                          }}
                          onClick={() => onSelectEvidence({ nodeType: 'evidence', ...sent })}
                          onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-1px)'}
                          onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                              <span style={{
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.75rem',
                                color: badgeColor,
                                fontWeight: 700
                              }}>
                                Sentence [{sent.sentence_idx}]
                              </span>
                              <span style={{
                                fontSize: '0.72rem',
                                fontFamily: 'var(--font-mono)',
                                color: badgeColor,
                                background: 'rgba(0,0,0,0.25)',
                                padding: '1px 6px',
                                borderRadius: '4px',
                                fontWeight: 600
                              }}>
                                {sent.entailment_label}
                              </span>

                              {sent.veto_masked && (
                                <span style={{
                                  fontSize: '0.68rem',
                                  fontFamily: 'var(--font-mono)',
                                  color: '#93c5fd',
                                  background: 'rgba(59, 130, 246, 0.18)',
                                  border: '1px solid rgba(59, 130, 246, 0.35)',
                                  padding: '1px 6px',
                                  borderRadius: '4px',
                                  fontWeight: 500
                                }}>
                                  Veto Excluded (Hill's Reversibility)
                                </span>
                              )}
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                              <span style={{ color: 'var(--text-muted)' }}>
                                Conf: <strong style={{ color: 'var(--text-primary)' }}>{Math.round(sent.confidence * 100)}%</strong>
                              </span>
                              <span style={{ color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '2px' }}>
                                Inspect <ChevronRight size={13} />
                              </span>
                            </div>
                          </div>

                          <div style={{
                            fontSize: '0.875rem',
                            color: '#f8fafc',
                            lineHeight: '1.5'
                          }}>
                            "{sent.text}"
                          </div>

                          {sent.veto_masked && sent.reversibility_rationale && (
                            <div style={{
                              display: 'flex',
                              alignItems: 'flex-start',
                              gap: '0.4rem',
                              fontSize: '0.75rem',
                              color: '#93c5fd',
                              background: 'rgba(15, 23, 42, 0.5)',
                              border: '1px solid rgba(59, 130, 246, 0.25)',
                              borderRadius: '6px',
                              padding: '0.4rem 0.65rem',
                              marginTop: '0.25rem'
                            }}>
                              <Info size={13} style={{ flexShrink: 0, marginTop: '2px', color: '#60a5fa' }} />
                              <span>{sent.reversibility_rationale}</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

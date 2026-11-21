import React, { useState, useEffect } from 'react';
import { X, FileText, Loader2, AlertCircle, Bookmark } from 'lucide-react';

export default function PaperModal({ paperId, onClose, highlightedSentenceIndices = [] }) {
  const [paper, setPaper] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!paperId) return;

    let isMounted = true;
    setLoading(true);
    setError(null);

    fetch(`/api/papers/${paperId}`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to load paper #${paperId} (Status ${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        if (isMounted) {
          setPaper(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      });

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      isMounted = false;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [paperId, onClose]);

  if (!paperId) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(9, 13, 22, 0.85)',
        backdropFilter: 'blur(8px)',
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '16px',
          width: '100%',
          maxWidth: '750px',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 50px rgba(0, 0, 0, 0.7)',
          overflow: 'hidden'
        }}
      >
        {/* Modal Header */}
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '1rem',
          background: 'rgba(15, 23, 42, 0.8)'
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
                background: 'rgba(56, 189, 248, 0.12)',
                color: 'var(--accent-cyan)',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                padding: '2px 8px',
                borderRadius: '4px',
                fontWeight: 600
              }}>
                Doc #{paperId}
              </span>
              {paper?.structured && (
                <span style={{
                  background: 'rgba(99, 102, 241, 0.12)',
                  color: '#a5b4fc',
                  border: '1px solid rgba(99, 102, 241, 0.3)',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  Structured Abstract
                </span>
              )}
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: '1.4' }}>
              {paper ? paper.title : `Loading Paper #${paperId}...`}
            </h3>
          </div>

          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0
            }}
            onMouseEnter={(e) => e.currentTarget.style.color = '#fff'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '1.5rem', overflowY: 'auto', flex: 1 }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '3rem', color: 'var(--accent-cyan)', gap: '0.75rem' }}>
              <Loader2 size={24} className="animate-spin" />
              <span>Fetching paper abstract from corpus...</span>
            </div>
          )}

          {error && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '1rem',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              borderRadius: '8px',
              color: '#fb7185'
            }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {paper && (
            <div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '1rem',
                fontSize: '0.8rem',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)'
              }}>
                <span>SENTENCE-BY-SENTENCE ABSTRACT VIEW</span>
                <span>{paper.abstract?.length || 0} sentences</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {paper.abstract?.map((sentText, sIdx) => {
                  const isHighlighted = highlightedSentenceIndices.includes(sIdx);

                  return (
                    <div
                      key={sIdx}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.75rem',
                        padding: '0.75rem 1rem',
                        borderRadius: '8px',
                        background: isHighlighted ? 'rgba(56, 189, 248, 0.12)' : 'rgba(15, 23, 42, 0.4)',
                        border: isHighlighted ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
                        transition: 'background 0.15s ease'
                      }}
                    >
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        color: isHighlighted ? 'var(--accent-cyan)' : 'var(--text-muted)',
                        paddingTop: '2px',
                        flexShrink: 0
                      }}>
                        [{sIdx}]
                      </span>
                      <p style={{
                        fontSize: '0.9rem',
                        lineHeight: '1.6',
                        color: isHighlighted ? '#f8fafc' : 'var(--text-secondary)'
                      }}>
                        {sentText}
                      </p>
                      {isHighlighted && (
                        <span style={{
                          marginLeft: 'auto',
                          background: 'rgba(56, 189, 248, 0.2)',
                          color: 'var(--accent-cyan)',
                          fontSize: '0.7rem',
                          fontFamily: 'var(--font-mono)',
                          padding: '1px 6px',
                          borderRadius: '4px',
                          whiteSpace: 'nowrap'
                        }}>
                          Candidate Evidence
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div style={{
          padding: '1rem 1.5rem',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(15, 23, 42, 0.8)',
          fontSize: '0.8rem',
          color: 'var(--text-muted)'
        }}>
          <span>Corpus Source: SciFact Abstract Collection (5,183 articles)</span>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255, 255, 255, 0.08)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
              padding: '0.45rem 1rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              cursor: 'pointer'
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

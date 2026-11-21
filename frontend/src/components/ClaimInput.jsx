import React, { useState, useEffect } from 'react';
import { Search, BookOpen, SlidersHorizontal, Loader2, ArrowRight } from 'lucide-react';

export default function ClaimInput({ onVerify, onAnalyze, loading, isLoading, initialClaim }) {
  const [claim, setClaim] = useState(initialClaim || '');
  const [samples, setSamples] = useState([]);
  const [showConfig, setShowConfig] = useState(false);
  const [topKDocs, setTopKDocs] = useState(5);
  const [topKSentences, setTopKSentences] = useState(3);

  useEffect(() => {
    if (initialClaim) {
      setClaim(initialClaim);
    }
  }, [initialClaim]);

  const isBusy = loading || isLoading;

  const triggerAnalyze = (claimText, docs = topKDocs, sentences = topKSentences, claimId = null) => {
    if (onVerify) {
      onVerify(claimText, docs, sentences, claimId);
    } else if (onAnalyze) {
      onAnalyze({ claim: claimText, top_k_docs: docs, top_k_sentences: sentences, claim_id: claimId });
    }
  };

  // Fetch real sample claims from backend
  useEffect(() => {
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        const sampleList = data.samples || (Array.isArray(data) ? data : []);
        if (sampleList && sampleList.length > 0) {
          setSamples(sampleList);
        }
      })
      .catch(() => {
        // Fallback real SciFact claims with verified gold ground truth across all 3 classes
        setSamples([
          { id: 49, claim: 'ADAR1 binds to Dicer to cleave pre-miRNA.', gold_label: 'SUPPORTS' },
          { id: 50, claim: 'AIRE is expressed in some skin tumors.', gold_label: 'SUPPORTS' },
          { id: 124, claim: 'Antiretroviral therapy reduces rates of tuberculosis across a broad range of CD4 strata.', gold_label: 'SUPPORTS' },
          { id: 51, claim: 'ALDH1 expression is associated with better breast cancer outcomes.', gold_label: 'CONTRADICTS' },
          { id: 42, claim: 'A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.', gold_label: 'CONTRADICTS' },
          { id: 338, claim: 'Dexamethasone decreases risk of postoperative bleeding.', gold_label: 'CONTRADICTS' },
          { id: 1, claim: '0-dimensional biomaterials show inductive properties.', gold_label: 'NOT ENOUGH INFORMATION' },
          { id: 13, claim: '5% of perinatal mortality is due to low birth weight.', gold_label: 'NOT ENOUGH INFORMATION' },
          { id: 36, claim: 'A deficiency of vitamin B12 increases blood levels of homocysteine.', gold_label: 'NOT ENOUGH INFORMATION' },
        ]);
      });
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!claim.trim() || isBusy) return;
    triggerAnalyze(claim.trim(), topKDocs, topKSentences);
  };

  const handleSelectSample = (sampleItem) => {
    if (typeof sampleItem === 'string') {
      setClaim(sampleItem);
      triggerAnalyze(sampleItem, topKDocs, topKSentences);
    } else {
      setClaim(sampleItem.claim);
      triggerAnalyze(sampleItem.claim, topKDocs, topKSentences, sampleItem.id);
    }
  };

  return (
    <div className="academic-card" style={{ marginBottom: '1.5rem' }}>
      <form onSubmit={handleSubmit}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Search size={16} color="var(--accent-cyan)" />
            Scientific Claim
          </label>
          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            style={{
              background: 'transparent',
              border: 'none',
              color: showConfig ? 'var(--accent-cyan)' : 'var(--text-muted)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
          >
            <SlidersHorizontal size={14} />
            <span>Retrieval Settings {showConfig ? '▲' : '▼'}</span>
          </button>
        </div>

        <div style={{ position: 'relative', marginBottom: '1rem' }}>
          <textarea
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            placeholder="Enter a scientific claim to verify (e.g. 'Smoking increases the risk of developing lung cancer.')..."
            rows={3}
            style={{
              width: '100%',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '0.875rem 1rem',
              color: 'var(--text-primary)',
              fontSize: '0.95rem',
              fontFamily: 'var(--font-sans)',
              resize: 'vertical',
              outline: 'none',
              transition: 'border-color 0.2s',
            }}
            onFocus={(e) => (e.target.style.borderColor = 'var(--border-focus)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--border-subtle)')}
          />
        </div>

        {showConfig && (
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '1rem',
              marginBottom: '1rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '1.25rem',
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Top Candidate Papers (k):</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{topKDocs}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                value={topKDocs}
                onChange={(e) => setTopKDocs(parseInt(e.target.value, 10))}
                style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Candidate Sentences per Paper:</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{topKSentences}</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                value={topKSentences}
                onChange={(e) => setTopKSentences(parseInt(e.target.value, 10))}
                style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
              />
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            BM25 Lexical Retrieval + DeBERTa-v3 NLI Inference
          </div>

          <button
            type="submit"
            disabled={!claim.trim() || isBusy}
            style={{
              background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              padding: '0.65rem 1.4rem',
              fontWeight: 600,
              fontSize: '0.875rem',
              cursor: !claim.trim() || isBusy ? 'not-allowed' : 'pointer',
              opacity: !claim.trim() || isBusy ? 0.6 : 1,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'opacity 0.2s',
            }}
          >
            {isBusy ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                <span>Verifying...</span>
              </>
            ) : (
              <>
                <span>Verify Claim</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Real Sample Claims from SciFact */}
      {samples.length > 0 && (
        <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <BookOpen size={13} color="var(--accent-cyan)" />
            <span>Sample SciFact Claims (click to evaluate):</span>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {samples.map((s, idx) => {
              const labelColor =
                s.gold_label === 'SUPPORTS'
                  ? 'var(--color-supports-border)'
                  : s.gold_label === 'CONTRADICTS'
                  ? 'var(--color-contradicts-border)'
                  : 'var(--color-nei-border)';

              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleSelectSample(s)}
                  style={{
                    background: 'var(--bg-secondary)',
                    border: `1px solid ${labelColor}`,
                    borderRadius: '6px',
                    padding: '0.4rem 0.75rem',
                    color: 'var(--text-secondary)',
                    fontSize: '0.78rem',
                    textAlign: 'left',
                    cursor: 'pointer',
                    maxWidth: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = 'var(--text-primary)';
                    e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = 'var(--text-secondary)';
                    e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontFamily: 'var(--font-mono)',
                      padding: '1px 5px',
                      borderRadius: '4px',
                      backgroundColor:
                        s.gold_label === 'SUPPORTS'
                          ? 'var(--color-supports-bg)'
                          : s.gold_label === 'CONTRADICTS'
                          ? 'var(--color-contradicts-bg)'
                          : 'var(--color-nei-bg)',
                      color:
                        s.gold_label === 'SUPPORTS'
                          ? 'var(--color-supports)'
                          : s.gold_label === 'CONTRADICTS'
                          ? 'var(--color-contradicts)'
                          : 'var(--color-nei)',
                    }}
                  >
                    {s.gold_label}
                  </span>
                  <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                    {s.claim}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

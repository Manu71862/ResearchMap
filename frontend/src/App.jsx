import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ClaimInput from './components/ClaimInput';
import VerdictCard from './components/VerdictCard';
import EvidenceGraph from './components/EvidenceGraph';
import EvidenceList from './components/EvidenceList';
import NodeDetailDrawer from './components/NodeDetailDrawer';
import PaperModal from './components/PaperModal';
import BenchmarkSection from './components/BenchmarkSection';
import {
  Loader2,
  AlertCircle,
  Database,
  Cpu,
  GitFork,
  ShieldCheck
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('explorer'); // 'explorer' | 'benchmarks'
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Inspector drawer state
  const [selectedNode, setSelectedNode] = useState(null);

  // Paper modal state
  const [modalPaperId, setModalPaperId] = useState(null);
  const [currentClaim, setCurrentClaim] = useState('');

  const handleVerify = async (claimText, topKDocs = 5, topKSentences = 3, claimId = null) => {
    setCurrentClaim(claimText);
    setLoading(true);
    setResult(null); // Prevent rendering stale results during in-flight analysis
    setError(null);
    setSelectedNode(null);

    try {
      const payload = {
        claim: claimText,
        top_k_docs: topKDocs,
        top_k_sentences: topKSentences,
      };
      if (claimId !== null && claimId !== undefined) {
        payload.claim_id = claimId;
      }

      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let errMessage = `Verification failed (Status ${response.status})`;
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errData = await response.json();
            if (errData.detail) errMessage = errData.detail;
          } else {
            const text = await response.text();
            if (text && text.trim().length > 0 && text.length < 300) {
              errMessage = `${errMessage}: ${text.trim()}`;
            }
          }
        } catch (_) {}
        throw new Error(errMessage);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Error during verification:', err);
      setError(err.message || 'An unexpected error occurred while communicating with the ResearchMap backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const queryClaim = params.get('claim');
      if (queryClaim) {
        setCurrentClaim(queryClaim);
        handleVerify(queryClaim);
      }
    } catch (_) {}
  }, []);

  const handleSelectNode = (nodeData) => {
    setSelectedNode(nodeData);
  };

  const handleCloseDrawer = () => {
    setSelectedNode(null);
  };

  const handleViewPaper = (paperId) => {
    setModalPaperId(paperId);
  };

  const handleCloseModal = () => {
    setModalPaperId(null);
  };

  // Find highlighted sentence indices for the active paper modal
  const modalHighlightedIndices = result?.evidence_details
    ?.filter((e) => e.doc_id === modalPaperId)
    ?.map((e) => e.sentence_idx) || [];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-primary)' }}>
      {/* Navigation Bar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main style={{ flex: 1, padding: '2rem 1.5rem 4rem' }}>
        <div style={{ maxWidth: '1240px', margin: '0 auto' }}>

          {/* Hero Header Section */}
          {activeTab === 'explorer' && (
            <div className="hero-section">
              <h1 className="hero-title">
                ResearchMap Claim Explorer
              </h1>
              <p className="hero-subtitle">
                Verify scientific claims against peer-reviewed biomedical literature. Retrieves relevant abstracts
                via BM25 lexical ranking and evaluates evidence using a transformer-based natural language inference model.
              </p>

              {/* Architecture Badges */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '1rem',
                flexWrap: 'wrap',
                marginBottom: '1.5rem'
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  background: 'rgba(15, 23, 42, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  padding: '3px 10px',
                  borderRadius: '20px'
                }}>
                  <Database size={13} style={{ color: 'var(--accent-cyan)' }} />
                  <span>Corpus: SciFact (5,183 Abstracts)</span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  background: 'rgba(15, 23, 42, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  padding: '3px 10px',
                  borderRadius: '20px'
                }}>
                  <Cpu size={13} style={{ color: 'var(--accent-blue)' }} />
                  <span>Model: DeBERTa-v3 Cross-Encoder</span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  background: 'rgba(15, 23, 42, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  padding: '3px 10px',
                  borderRadius: '20px'
                }}>
                  <GitFork size={13} style={{ color: '#34d399' }} />
                  <span>Output: SUPPORTS / CONTRADICTS / NEI</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 1: Claim Explorer */}
          {activeTab === 'explorer' && (
            <div>
              {/* Claim Input Component */}
              <ClaimInput onVerify={handleVerify} isLoading={loading} initialClaim={currentClaim} />

              {/* Loading State Spinner */}
              {loading && (
                <div style={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '14px',
                  padding: '3.5rem 2rem',
                  textAlign: 'center',
                  marginBottom: '2rem',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '1rem'
                }}>
                  <div style={{
                    position: 'relative',
                    width: '4rem',
                    height: '4rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Loader2 size={40} className="animate-spin" style={{ color: 'var(--accent-cyan)' }} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>
                      Evaluating Claim Against Scientific Literature
                    </h3>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Retrieving candidate abstracts via BM25 → Ranking sentence evidence → Computing NLI entailment probabilities...
                    </p>
                  </div>
                </div>
              )}

              {/* Error Alert Box */}
              {error && (
                <div style={{
                  background: 'rgba(244, 63, 94, 0.1)',
                  border: '1px solid rgba(244, 63, 94, 0.3)',
                  borderRadius: '12px',
                  padding: '1.25rem 1.5rem',
                  marginBottom: '2rem',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1rem',
                  color: '#fb7185'
                }}>
                  <AlertCircle size={22} style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <strong style={{ fontSize: '0.95rem' }}>Verification Error</strong>
                    <p style={{ fontSize: '0.85rem', marginTop: '0.25rem', color: '#fecdd3' }}>{error}</p>
                  </div>
                </div>
              )}

              {/* Results Section */}
              {result && !loading && (
                <div>
                  {/* Verdict & Confidence Summary Card */}
                  <VerdictCard result={result} />

                  {/* Interactive Evidence Graph */}
                  <EvidenceGraph
                    result={result}
                    onSelectNode={handleSelectNode}
                  />

                  {/* Retrieved Papers & Passages List */}
                  <EvidenceList
                    result={result}
                    onSelectEvidence={handleSelectNode}
                    onViewPaper={handleViewPaper}
                  />
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Model Performance & Benchmarks */}
          {activeTab === 'benchmarks' && (
            <BenchmarkSection />
          )}

        </div>
      </main>

      {/* Side Detail Inspector Drawer */}
      <NodeDetailDrawer
        selectedNode={selectedNode}
        onClose={handleCloseDrawer}
        onViewPaper={handleViewPaper}
      />

      {/* Paper Abstract Modal */}
      <PaperModal
        paperId={modalPaperId}
        onClose={handleCloseModal}
        highlightedSentenceIndices={modalHighlightedIndices}
      />

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(15, 23, 42, 0.85)',
        padding: '1.75rem 1.5rem',
        marginTop: 'auto',
        fontSize: '0.8rem',
        color: 'var(--text-muted)'
      }}>
        <div style={{
          maxWidth: '1240px',
          margin: '0 auto',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem'
        }}>
          <div>
            <strong style={{ color: 'var(--text-secondary)' }}>ResearchMap</strong> — Scientific Claim Verification
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            <ShieldCheck size={14} style={{ color: '#10b981' }} />
            <span>Evaluated on SciFact benchmark • BM25 + DeBERTa-v3 NLI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

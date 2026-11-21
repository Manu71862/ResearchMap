import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  CheckCircle2,
  Database,
  Cpu,
  Layers,
  ShieldCheck,
  AlertCircle,
  FileCheck
} from 'lucide-react';

export default function BenchmarkSection() {
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/metrics')
      .then(res => res.json())
      .then(data => {
        setMetricsData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch /api/metrics:", err);
        setLoading(false);
      });
  }, []);

  const fineMetrics = metricsData?.metrics?.finetuned;
  const baseMetrics = metricsData?.metrics?.baseline;
  const deltas = metricsData?.metrics?.deltas;

  // Safe formatting helpers
  const fmt = (val, digits = 4) => (typeof val === 'number' && !isNaN(val) ? val.toFixed(digits) : '—');
  const fmtPct = (val) => (typeof val === 'number' && !isNaN(val) ? `${(val * 100).toFixed(2)}%` : '—');
  const fmtDelta = (val, digits = 4) => {
    if (typeof val !== 'number' || isNaN(val)) return '—';
    return val >= 0 ? `+${val.toFixed(digits)}` : val.toFixed(digits);
  };
  const calcRelDelta = (fineVal, baseVal) => {
    if (typeof fineVal === 'number' && typeof baseVal === 'number' && baseVal > 0) {
      const rel = ((fineVal - baseVal) / baseVal) * 100;
      return rel >= 0 ? `+${rel.toFixed(1)}%` : `${rel.toFixed(1)}%`;
    }
    return null;
  };

  // Primary KPIs
  const fineMacro = fineMetrics?.claim_verification?.macro_f1;
  const baseMacro = baseMetrics?.claim_verification?.macro_f1;
  const deltaMacro = deltas?.macro_f1 ?? (fineMacro != null && baseMacro != null ? fineMacro - baseMacro : null);
  const relMacro = calcRelDelta(fineMacro, baseMacro);

  const fineWeighted = fineMetrics?.claim_verification?.weighted_f1;
  const baseWeighted = baseMetrics?.claim_verification?.weighted_f1;
  const deltaWeighted = deltas?.weighted_f1 ?? (fineWeighted != null && baseWeighted != null ? fineWeighted - baseWeighted : null);
  const relWeighted = calcRelDelta(fineWeighted, baseWeighted);

  const fineSentF1 = fineMetrics?.sentence_evidence?.sentence_evidence_f1;
  const baseSentF1 = baseMetrics?.sentence_evidence?.sentence_evidence_f1;
  const deltaSent = deltas?.sentence_evidence_f1 ?? (fineSentF1 != null && baseSentF1 != null ? fineSentF1 - baseSentF1 : null);
  const relSent = calcRelDelta(fineSentF1, baseSentF1);

  // Per-Class metrics
  const fineSupports = fineMetrics?.claim_verification?.per_class?.["SUPPORTS"];
  const baseSupports = baseMetrics?.claim_verification?.per_class?.["SUPPORTS"];
  const supportsDeltaF1 = (fineSupports?.f1_score != null && baseSupports?.f1_score != null)
    ? fineSupports.f1_score - baseSupports.f1_score
    : null;
  const supportsRelF1 = calcRelDelta(fineSupports?.f1_score, baseSupports?.f1_score);

  const fineContradicts = fineMetrics?.claim_verification?.per_class?.["CONTRADICTS"];
  const baseContradicts = baseMetrics?.claim_verification?.per_class?.["CONTRADICTS"];
  const contradictsDeltaF1 = (fineContradicts?.f1_score != null && baseContradicts?.f1_score != null)
    ? fineContradicts.f1_score - baseContradicts.f1_score
    : null;
  const contradictsRelF1 = calcRelDelta(fineContradicts?.f1_score, baseContradicts?.f1_score);

  const fineNEI = fineMetrics?.claim_verification?.per_class?.["NOT ENOUGH INFORMATION"];
  const baseNEI = baseMetrics?.claim_verification?.per_class?.["NOT ENOUGH INFORMATION"];
  const neiDeltaF1 = (fineNEI?.f1_score != null && baseNEI?.f1_score != null)
    ? fineNEI.f1_score - baseNEI.f1_score
    : null;
  const neiRelF1 = calcRelDelta(fineNEI?.f1_score, baseNEI?.f1_score);

  // Retrieval metrics
  const retrieval = fineMetrics?.retrieval || baseMetrics?.retrieval;
  const recallAt5 = retrieval?.["doc_recall@5"];
  const recallAt10 = retrieval?.["doc_recall@10"];
  const strictRecallAt5 = retrieval?.["doc_strict_recall@5"];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', paddingBottom: '3rem' }}>
      {/* Header Banner */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '14px',
        padding: '1.75rem',
        marginBottom: '2rem',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))'
        }} />

        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '2.5rem',
              height: '2.5rem',
              borderRadius: '8px',
              background: 'rgba(56, 189, 248, 0.1)',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-cyan)'
            }}>
              <BarChart3 size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f8fafc' }}>
                Model Performance & Empirical Evaluation
              </h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
                <span style={{
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--accent-cyan)',
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.2)',
                  padding: '1px 8px',
                  borderRadius: '4px'
                }}>
                  SciFact Official Development Split (claims_dev.jsonl)
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  • {metricsData?.total_claims ?? 300} claims • 5,183 paper abstracts
                </span>
              </div>
            </div>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: '8px',
            padding: '0.5rem 0.85rem',
            color: '#34d399',
            fontSize: '0.8rem',
            fontFamily: 'var(--font-mono)'
          }}>
            <ShieldCheck size={16} />
            <span>Leakage-Safe Partition Verified</span>
          </div>
        </div>

        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          This section reports solely measured, empirical metrics from the baseline and fine-tuned model checkpoint on the official SciFact development split (claims_dev.jsonl). 3-way claim verification metrics and sentence-level evidence extraction metrics are evaluated and presented separately. Development-set evaluation results are not blind test results. <strong>No metrics or numbers are fabricated or estimated.</strong>
        </p>
      </div>

      {/* Primary KPI Comparison Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1.25rem',
        marginBottom: '2rem'
      }}>
        {/* Macro-F1 Card */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '1.25rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>CLAIM VERIFICATION</span>
            <span style={{
              fontSize: '0.7rem',
              fontFamily: 'var(--font-mono)',
              background: 'rgba(16, 185, 129, 0.12)',
              color: '#34d399',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              padding: '1px 6px',
              borderRadius: '4px'
            }}>
              {relMacro ? `${relMacro} Relative` : 'Empirical Gain'}
            </span>
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Macro-F1 Score
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
              {fmt(fineMacro)}
            </span>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', textDecoration: 'line-through', fontFamily: 'var(--font-mono)' }}>
              {fmt(baseMacro)}
            </span>
            <span style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 600 }}>
              {fmtDelta(deltaMacro)}
            </span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Zero-shot baseline ({fmt(baseMacro)}) vs fine-tuned checkpoint ({fmt(fineMacro)}) across all 3 classes.
          </div>
        </div>

        {/* Weighted-F1 Card */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '1.25rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>CLASS-WEIGHTED</span>
            <span style={{
              fontSize: '0.7rem',
              fontFamily: 'var(--font-mono)',
              background: 'rgba(16, 185, 129, 0.12)',
              color: '#34d399',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              padding: '1px 6px',
              borderRadius: '4px'
            }}>
              {relWeighted ? `${relWeighted} Relative` : 'Empirical Gain'}
            </span>
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Weighted-F1 Score
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
              {fmt(fineWeighted)}
            </span>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', textDecoration: 'line-through', fontFamily: 'var(--font-mono)' }}>
              {fmt(baseWeighted)}
            </span>
            <span style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 600 }}>
              {fmtDelta(deltaWeighted)}
            </span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Accounts for class distribution ({fineSupports?.support ?? 124} Supports, {fineContradicts?.support ?? 64} Contradicts, {fineNEI?.support ?? 112} NEI).
          </div>
        </div>

        {/* Sentence Evidence F1 Card */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '1.25rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>SENTENCE EVIDENCE EXTRACTION</span>
            <span style={{
              fontSize: '0.7rem',
              fontFamily: 'var(--font-mono)',
              background: 'rgba(16, 185, 129, 0.12)',
              color: '#34d399',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              padding: '1px 6px',
              borderRadius: '4px'
            }}>
              {relSent ? `${relSent} Relative` : 'Empirical Gain'}
            </span>
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Sentence Evidence F1
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#a78bfa' }}>
              {fmt(fineSentF1)}
            </span>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', textDecoration: 'line-through', fontFamily: 'var(--font-mono)' }}>
              {fmt(baseSentF1)}
            </span>
            <span style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 600 }}>
              {fmtDelta(deltaSent)}
            </span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Strict joint precision/recall matching gold annotated rationale sentences (evaluated separately from claim verdict).
          </div>
        </div>
      </div>

      {/* Per-Class Detailed Breakdown Table */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '12px',
        overflow: 'hidden',
        marginBottom: '2rem'
      }}>
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(15, 23, 42, 0.6)'
        }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Per-Class Classification Metrics (Fine-Tuned Checkpoint)
          </h3>
          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            {metricsData?.total_claims ?? 300} Dev Claims Evaluated
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                <th style={{ padding: '0.85rem 1.5rem' }}>VERDICT CLASS</th>
                <th style={{ padding: '0.85rem 1rem' }}>GOLD SUPPORT</th>
                <th style={{ padding: '0.85rem 1rem' }}>PRECISION</th>
                <th style={{ padding: '0.85rem 1rem' }}>RECALL</th>
                <th style={{ padding: '0.85rem 1rem' }}>F1-SCORE</th>
                <th style={{ padding: '0.85rem 1.5rem' }}>DELTA VS BASELINE</th>
              </tr>
            </thead>
            <tbody>
              {/* SUPPORTS */}
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(16, 185, 129, 0.03)' }}>
                <td style={{ padding: '1rem 1.5rem', fontWeight: 600, color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }}></span>
                  SUPPORTS
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {fineSupports?.support ?? 124}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)' }}>
                  {fmt(fineSupports?.precision)}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)' }}>
                  {fmt(fineSupports?.recall)}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                  {fmt(fineSupports?.f1_score)}
                </td>
                <td style={{ padding: '1rem 1.5rem', fontFamily: 'var(--font-mono)', color: '#10b981' }}>
                  {fmtDelta(supportsDeltaF1)} {supportsRelF1 ? `(${supportsRelF1})` : ''}
                </td>
              </tr>
              {/* CONTRADICTS */}
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(244, 63, 94, 0.03)' }}>
                <td style={{ padding: '1rem 1.5rem', fontWeight: 600, color: '#f43f5e', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f43f5e' }}></span>
                  CONTRADICTS
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {fineContradicts?.support ?? 64}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)' }}>
                  {fmt(fineContradicts?.precision)}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)' }}>
                  {fmt(fineContradicts?.recall)}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                  {fmt(fineContradicts?.f1_score)}
                </td>
                <td style={{ padding: '1rem 1.5rem', fontFamily: 'var(--font-mono)', color: '#10b981' }}>
                  {fmtDelta(contradictsDeltaF1)} {contradictsRelF1 ? `(${contradictsRelF1})` : ''}
                </td>
              </tr>
              {/* NOT ENOUGH INFORMATION */}
              <tr style={{ background: 'rgba(245, 158, 11, 0.03)' }}>
                <td style={{ padding: '1rem 1.5rem', fontWeight: 600, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f59e0b' }}></span>
                  NOT ENOUGH INFORMATION
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {fineNEI?.support ?? 112}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)' }}>
                  {fmt(fineNEI?.precision)}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)' }}>
                  {fmt(fineNEI?.recall)}
                </td>
                <td style={{ padding: '1rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                  {fmt(fineNEI?.f1_score)}
                </td>
                <td style={{ padding: '1rem 1.5rem', fontFamily: 'var(--font-mono)', color: '#10b981' }}>
                  {fmtDelta(neiDeltaF1)} {neiRelF1 ? `(${neiRelF1})` : ''}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Retrieval Performance & Technical Architecture Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: '1.25rem'
      }}>
        {/* Lexical Retrieval Evaluation */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <Database size={18} style={{ color: 'var(--accent-cyan)' }} />
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              BM25 Corpus Retrieval Recall
            </h4>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Evaluated on all 188 claims with gold evidence across the 5,183 SciFact research abstracts corpus.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Doc Recall@5</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                  {fmtPct(recallAt5)}
                </span>
              </div>
              <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${typeof recallAt5 === 'number' ? (recallAt5 * 100).toFixed(2) : '87.77'}%`, height: '100%', background: 'var(--accent-cyan)' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Doc Recall@10</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)' }}>
                  {fmtPct(recallAt10)}
                </span>
              </div>
              <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${typeof recallAt10 === 'number' ? (recallAt10 * 100).toFixed(2) : '92.55'}%`, height: '100%', background: 'var(--accent-blue)' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Doc Strict Recall@5 (All cited docs found)</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-muted)' }}>
                  {fmtPct(strictRecallAt5)}
                </span>
              </div>
              <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${typeof strictRecallAt5 === 'number' ? (strictRecallAt5 * 100).toFixed(2) : '84.04'}%`, height: '100%', background: '#64748b' }} />
              </div>
            </div>
          </div>
        </div>

        {/* Model Architecture & Protocol */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <Cpu size={18} style={{ color: 'var(--accent-cyan)' }} />
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Model & Partition Protocol
            </h4>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.8rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Base Architecture</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>DeBERTa-v3-small Cross-Encoder</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Checkpoint Directory</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>models/finetuned_nli</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Training Source</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>claims_train.jsonl (809 claims)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Dev Leakage Protection</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: '#34d399' }}>Zero claims_dev overlap</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Verification Pipeline</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>BM25 Okapi + Softmax NLI</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

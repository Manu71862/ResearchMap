import React from 'react';
import {
  CheckCircle2,
  XCircle,
  HelpCircle,
  AlertTriangle,
  ShieldCheck,
  BarChart3,
  FileText,
  Layers
} from 'lucide-react';

export default function VerdictCard({ result }) {
  if (!result) return null;

  const {
    predicted_label,
    confidence,
    calibrated_probabilities = {},
    warning,
    explanation,
    disclaimer,
    retrieved_papers = [],
    evidence_details = [],
    predicted_evidence_doc_ids = []
  } = result;

  // Determine color scheme based on label
  const isSupports = predicted_label === 'SUPPORTS';
  const isContradicts = predicted_label === 'CONTRADICTS';
  const isNEI = predicted_label === 'NOT ENOUGH INFORMATION';

  const badgeConfig = isSupports
    ? {
        label: 'SUPPORTS',
        icon: CheckCircle2,
        colorClass: 'text-emerald-400',
        bgClass: 'bg-emerald-950/40 border-emerald-500/50',
        glowClass: 'shadow-[0_0_25px_rgba(16,185,129,0.2)]',
        accentColor: '#10b981',
        description: 'Scientific evidence retrieved from peer-reviewed literature supports this claim.'
      }
    : isContradicts
    ? {
        label: 'CONTRADICTS',
        icon: XCircle,
        colorClass: 'text-rose-400',
        bgClass: 'bg-rose-950/40 border-rose-500/50',
        glowClass: 'shadow-[0_0_25px_rgba(244,63,94,0.2)]',
        accentColor: '#f43f5e',
        description: 'Scientific evidence retrieved from peer-reviewed literature refutes or contradicts this claim.'
      }
    : {
        label: 'NOT ENOUGH INFORMATION',
        icon: HelpCircle,
        colorClass: 'text-amber-400',
        bgClass: 'bg-amber-950/40 border-amber-500/50',
        glowClass: 'shadow-[0_0_25px_rgba(245,158,11,0.2)]',
        accentColor: '#f59e0b',
        description: 'Retrieved literature does not provide conclusive evidence to support or refute this claim.'
      };

  const Icon = badgeConfig.icon;
  const pctConfidence = Math.round((confidence || 0) * 100);

  const supportsProb = Math.round((calibrated_probabilities.SUPPORTS || 0) * 100);
  const contradictsProb = Math.round((calibrated_probabilities.CONTRADICTS || 0) * 100);
  const neiProb = Math.round((calibrated_probabilities['NOT ENOUGH INFORMATION'] || 0) * 100);

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: `1px solid ${isSupports ? 'var(--color-supports-border)' : isContradicts ? 'var(--color-contradicts-border)' : 'var(--color-nei-border)'}`,
      borderRadius: '14px',
      padding: '1.75rem',
      marginBottom: '2rem',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: isSupports
        ? '0 8px 30px rgba(16, 185, 129, 0.08)'
        : isContradicts
        ? '0 8px 30px rgba(244, 63, 94, 0.08)'
        : '0 8px 30px rgba(245, 158, 11, 0.08)'
    }}>
      {/* Top Accent Stripe */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '4px',
        background: `linear-gradient(90deg, ${badgeConfig.accentColor}, transparent)`
      }} />

      {/* Main Header Row */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1.25rem',
        marginBottom: '1.5rem',
        paddingBottom: '1.25rem',
        borderBottom: '1px solid var(--border-subtle)'
      }}>
        {/* Left: Verdict Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '3.25rem',
            height: '3.25rem',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: isSupports
              ? 'var(--color-supports-bg)'
              : isContradicts
              ? 'var(--color-contradicts-bg)'
              : 'var(--color-nei-bg)',
            border: `1px solid ${isSupports ? 'var(--color-supports-border)' : isContradicts ? 'var(--color-contradicts-border)' : 'var(--color-nei-border)'}`,
            color: badgeConfig.accentColor
          }}>
            <Icon size={28} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>
              Model Verdict
            </div>
            <div style={{
              fontSize: '1.5rem',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              color: badgeConfig.accentColor
            }}>
              {badgeConfig.label}
            </div>
            {result.gold_label && (
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                marginTop: '0.35rem',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                padding: '2px 8px',
                borderRadius: '4px',
                backgroundColor: result.label_matches_gold ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                border: `1px solid ${result.label_matches_gold ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'}`,
                color: result.label_matches_gold ? '#34d399' : '#fb7185'
              }}>
                <span>SciFact Gold: <strong>{result.gold_label}</strong></span>
                <span>•</span>
                <span>{result.label_matches_gold ? '✓ Agreement' : '⚠ Benchmark Divergence'}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right: Confidence Metric Card */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1.5rem',
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '10px',
          padding: '0.75rem 1.25rem'
        }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>CONFIDENCE</div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {pctConfidence}%
            </div>
          </div>
          {/* Quick Stats Pill */}
          <div style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '1.25rem', display: 'flex', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Papers Retr.</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>{retrieved_papers.length}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Evidence Sents</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>{evidence_details.length}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Probability Distribution Breakdown */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <BarChart3 size={14} /> Class Probabilities Distribution
          </span>
          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            Raw Softmax Class Probabilities (Uncalibrated)
          </span>
        </div>

        {/* Segmented Bar */}
        <div style={{
          height: '10px',
          width: '100%',
          backgroundColor: '#1e293b',
          borderRadius: '9999px',
          overflow: 'hidden',
          display: 'flex'
        }}>
          <div style={{ width: `${supportsProb}%`, backgroundColor: '#10b981', transition: 'width 0.4s ease' }} title={`SUPPORTS: ${supportsProb}%`} />
          <div style={{ width: `${contradictsProb}%`, backgroundColor: '#f43f5e', transition: 'width 0.4s ease' }} title={`CONTRADICTS: ${contradictsProb}%`} />
          <div style={{ width: `${neiProb}%`, backgroundColor: '#f59e0b', transition: 'width 0.4s ease' }} title={`NOT ENOUGH INFO: ${neiProb}%`} />
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.6rem', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#10b981' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981' }}></span>
            <span>Supports: {supportsProb}%</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#f43f5e' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f43f5e' }}></span>
            <span>Contradicts: {contradictsProb}%</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#f59e0b' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f59e0b' }}></span>
            <span>Not Enough Info: {neiProb}%</span>
          </div>
        </div>
      </div>

      {/* Explanation Text */}
      {explanation && (
        <div style={{
          fontSize: '0.9rem',
          lineHeight: '1.6',
          color: 'var(--text-secondary)',
          background: 'rgba(15, 23, 42, 0.4)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '0.875rem 1rem',
          marginBottom: '1rem'
        }}>
          <strong style={{ color: 'var(--text-primary)' }}>Pipeline Analysis: </strong>
          {explanation}
        </div>
      )}

      {/* Warning Box (if evidence is weak, ambiguous, or zero retrieval) */}
      {warning && (
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          marginBottom: '1rem',
          color: '#fbbf24',
          fontSize: '0.85rem'
        }}>
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Notice: </strong> {warning}
          </div>
        </div>
      )}

      {/* Scientific Fact vs Model Prediction Disclaimer */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        borderTop: '1px solid var(--border-subtle)',
        paddingTop: '0.75rem'
      }}>
        <ShieldCheck size={14} style={{ color: 'var(--accent-cyan)' }} />
        <span>
          <strong>Scientific Disclaimer: </strong>
          {disclaimer || "Model predictions reflect statistical NLP inference over retrieved abstracts, not peer-reviewed scientific consensus."}
        </span>
      </div>
    </div>
  );
}

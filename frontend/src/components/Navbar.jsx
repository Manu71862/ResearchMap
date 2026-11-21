import React, { useEffect, useState } from 'react';
import { Network, Activity, BarChart3, BookOpen, ShieldCheck } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const [health, setHealth] = useState({ loaded: false, status: 'connecting', papers: 0 });

  useEffect(() => {
    let isMounted = true;

    const checkHealth = () => {
      fetch('/api/health')
        .then((res) => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          return res.json();
        })
        .then((data) => {
          if (isMounted) {
            setHealth({
              loaded: Boolean(data.model_loaded),
              status: data.status || 'healthy',
              papers: data.corpus_papers_count || 0,
            });
          }
        })
        .catch(() => {
          if (isMounted) {
            setHealth((prev) => ({ ...prev, loaded: false, status: 'offline' }));
          }
        });
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const getStatusLabel = () => {
    if (health.loaded) return 'Fine-Tuned Model Active';
    if (health.status === 'offline') return 'Backend Offline';
    if (health.status === 'degraded') return 'Model Loading...';
    return 'Connecting...';
  };

  const getStatusColor = () => {
    if (health.loaded) return 'var(--color-supports)';
    if (health.status === 'degraded') return 'var(--color-nei)';
    return '#ef4444';
  };

  return (
    <header className="navbar">
      <div className="container navbar-inner">
        <div className="brand-group">
          <div className="brand-icon">
            <Network size={20} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-title">ResearchMap</span>
              <span className="brand-tag">v1.0 Capstone</span>
            </div>
            <p style={{ fontSize: '0.725rem', color: 'var(--text-muted)', margin: 0 }}>
              Evidence Mapping & Scientific Claim Verification
            </p>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-btn ${activeTab === 'explorer' ? 'active' : ''}`}
            onClick={() => setActiveTab('explorer')}
          >
            <Activity size={16} />
            <span>Claim Explorer</span>
          </button>
          <button
            className={`nav-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarks')}
          >
            <BarChart3 size={16} />
            <span>Benchmark Reports</span>
          </button>

          <div className="system-status" title={`Corpus: ${health.papers.toLocaleString()} scientific papers indexed`}>
            <span
              className="status-dot"
              style={{
                background: getStatusColor(),
                boxShadow: `0 0 8px ${getStatusColor()}`,
              }}
            />
            <span>{getStatusLabel()}</span>
          </div>
        </nav>
      </div>
    </header>
  );
}

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Target, AlertTriangle, CheckCircle } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

export default function ScorePanel({ jobId }) {
  const [scoreData, setScoreData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchScore = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await api.get(`/jobs/${jobId}/score`);
        if (!cancelled) {
          if (res.data.error) {
            setError(res.data.error);
          } else {
            setScoreData(res.data);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError("Failed to fetch match score. Ensure Company Profile is set.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchScore();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (loading) return <div className="empty-state">Evaluating Match Score with LLM...</div>;
  if (error) return <div className="empty-state" style={{ color: 'var(--status-red)' }}>{error}</div>;
  if (!scoreData) return null;

  const getFeasibilityColor = (level) => {
    if (level === 'High') return 'var(--status-green)';
    if (level === 'Medium') return 'var(--status-yellow)';
    return 'var(--status-red)';
  };

  return (
    <div className="panel" style={{ padding: '24px', maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <Target size={28} color="var(--primary)" />
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600, color: 'var(--text)' }}>
          Tender Match Analysis
        </h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
        <div style={{ background: 'var(--surface-light)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Overall Match Score
          </div>
          <div style={{ fontSize: '3rem', fontWeight: 700, color: 'var(--text)', display: 'flex', alignItems: 'baseline', gap: '4px' }}>
            {scoreData.match_score}
            <span style={{ fontSize: '1.2rem', color: 'var(--text-muted)' }}>/ 100</span>
          </div>
          <div className="progress-track" style={{ marginTop: '12px', height: '6px' }}>
            <div 
              className="progress-fill" 
              style={{ 
                width: `${Math.max(0, Math.min(100, scoreData.match_score))}%`,
                background: scoreData.match_score > 70 ? 'var(--status-green)' : scoreData.match_score > 40 ? 'var(--status-yellow)' : 'var(--status-red)'
              }} 
            />
          </div>
        </div>

        <div style={{ background: 'var(--surface-light)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Cost Feasibility
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.5rem', fontWeight: 600, color: getFeasibilityColor(scoreData.cost_feasibility) }}>
            {scoreData.cost_feasibility === 'High' && <CheckCircle size={24} />}
            {scoreData.cost_feasibility === 'Medium' && <AlertTriangle size={24} />}
            {scoreData.cost_feasibility === 'Low' && <XCircle size={24} />}
            {scoreData.cost_feasibility}
          </div>
        </div>
      </div>

      <div style={{ background: 'var(--surface-light)', padding: '24px', borderRadius: '12px', border: '1px solid var(--border)' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', color: 'var(--text)' }}>Strategic Reasoning</h3>
        <p style={{ margin: 0, color: 'var(--text-muted)', lineHeight: '1.6' }}>
          {scoreData.reasoning}
        </p>
      </div>
    </div>
  );
}

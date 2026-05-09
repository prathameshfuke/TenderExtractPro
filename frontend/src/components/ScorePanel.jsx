import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Target, AlertTriangle, CheckCircle, Info, XCircle } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

export default function ScorePanel({ jobId, initialData }) {
  const [scoreData, setScoreData] = useState(initialData || null);
  const [loading, setLoading] = useState(!initialData);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (initialData) {
      setScoreData(initialData);
      setLoading(false);
      return;
    }

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
  }, [jobId, initialData]);

  if (loading) return <div className="empty-state">Evaluating Match Score with LLM...</div>;
  if (error) return (
    <div className="empty-state">
      <div style={{ color: 'var(--error)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
        <Info size={48} opacity={0.5} />
        <p style={{ maxWidth: '300px' }}>{error}</p>
      </div>
    </div>
  );
  if (!scoreData) return null;

  const getFeasibilityColor = (level) => {
    if (level === 'High') return '#16a34a';
    if (level === 'Medium') return '#d97706';
    return '#dc2626';
  };

  const score = scoreData.match_score || 0;
  const scoreColor = score > 70 ? '#16a34a' : score > 40 ? '#d97706' : '#dc2626';

  return (
    <div className="panel" style={{ padding: '48px', maxWidth: '1000px', margin: '0 auto', animation: 'fadeIn 0.4s ease-out' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '40px' }}>
        <div style={{ background: 'var(--surface-strong)', padding: '12px', borderRadius: '12px' }}>
          <Target size={32} color="var(--primary-ink)" />
        </div>
        <div>
          <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 300, color: 'var(--text-ink)', letterSpacing: '-0.02em' }}>
            Tender Match Analysis
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>
            AI-powered strategic alignment evaluation
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '32px', marginBottom: '40px' }}>
        <div style={{ background: 'var(--surface-card)', padding: '32px', borderRadius: '24px', border: '1px solid var(--hairline)', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
            Overall Match Score
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '4.5rem', fontWeight: 300, color: 'var(--text-ink)', lineHeight: 1 }}>
              {score}
            </span>
            <span style={{ fontSize: '1.5rem', color: 'var(--text-muted-soft)', fontWeight: 300 }}>/ 100</span>
          </div>
          <div className="progress-track" style={{ marginTop: '32px', height: '8px', background: 'var(--surface-strong)', borderRadius: '4px' }}>
            <div 
              className="progress-fill" 
              style={{ 
                width: `${Math.max(2, score)}%`,
                background: scoreColor,
                boxShadow: `0 0 12px ${scoreColor}44`
              }} 
            />
          </div>
          <div style={{ marginTop: '16px', fontSize: '13px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
             {score > 70 ? 'Strong strategic alignment detected.' : score > 40 ? 'Moderate alignment. Review specifics.' : 'Weak alignment. Proceed with caution.'}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ background: 'var(--surface-card)', padding: '24px', borderRadius: '20px', border: '1px solid var(--hairline)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '4px' }}>
                Cost Feasibility
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 500, color: getFeasibilityColor(scoreData.cost_feasibility) }}>
                {scoreData.cost_feasibility}
              </div>
            </div>
            <div style={{ color: getFeasibilityColor(scoreData.cost_feasibility), opacity: 0.8 }}>
              {scoreData.cost_feasibility === 'High' && <CheckCircle size={32} />}
              {scoreData.cost_feasibility === 'Medium' && <AlertTriangle size={32} />}
              {scoreData.cost_feasibility === 'Low' && <XCircle size={32} />}
            </div>
          </div>

          <div style={{ background: 'var(--primary-ink)', color: 'var(--canvas)', padding: '24px', borderRadius: '20px', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '4px' }}>
                Strategic Priority
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 500 }}>
                {score > 80 ? 'P0 - Direct Match' : score > 60 ? 'P1 - Target' : 'P2 - Opportunistic'}
              </div>
            </div>
            <div style={{ position: 'absolute', right: '-10px', bottom: '-10px', opacity: 0.1 }}>
              <Target size={100} />
            </div>
          </div>
        </div>
      </div>

      <div style={{ background: 'var(--surface-card)', padding: '32px', borderRadius: '24px', border: '1px solid var(--hairline)' }}>
        <h3 style={{ margin: '0 0 20px 0', fontSize: '1.25rem', fontWeight: 400, color: 'var(--text-ink)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Info size={20} color="var(--text-muted)" />
          Strategic Reasoning
        </h3>
        <p style={{ margin: 0, color: 'var(--text-body)', lineHeight: '1.8', fontSize: '16px' }}>
          {scoreData.reasoning}
        </p>
      </div>
    </div>
  );
}

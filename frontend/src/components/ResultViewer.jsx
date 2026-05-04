import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { XCircle } from 'lucide-react';
import SpecsTable from './SpecsTable';
import ScopePanel from './ScopePanel';
import ChatPanel from './ChatPanel';
import ScorePanel from './ScorePanel';

const api = axios.create({ baseURL: '/api' });

function confidenceToNumber(raw) {
  if (typeof raw === 'number') {
    return Math.max(0, Math.min(1, raw));
  }
  const normalized = String(raw || '').toUpperCase();
  if (normalized === 'HIGH') return 0.9;
  if (normalized === 'MEDIUM') return 0.65;
  if (normalized === 'LOW') return 0.35;
  return 0.0;
}

function normalizeSpecs(items) {
  return (Array.isArray(items) ? items : []).map((spec, index) => {
    const component = spec.component || spec.item_name || `Spec ${index + 1}`;

    let specsMap = {};
    if (spec.specs && typeof spec.specs === 'object' && !Array.isArray(spec.specs)) {
      specsMap = spec.specs;
    } else {
      const legacy = {
        specification_text: spec.specification_text,
        unit: spec.unit,
        numeric_value: spec.numeric_value,
        tolerance: spec.tolerance,
        standard_reference: spec.standard_reference,
        material: spec.material,
      };
      Object.entries(legacy).forEach(([key, value]) => {
        if (value && value !== 'NOT_FOUND') {
          specsMap[key] = String(value);
        }
      });
    }

    return {
      component,
      specs: specsMap,
      confidence: confidenceToNumber(spec.confidence),
      source: {
        chunk_id: spec?.source?.chunk_id || 'NOT_FOUND',
        page: spec?.source?.page || 0,
        clause: spec?.source?.clause || 'NOT_FOUND',
        exact_text: spec?.source?.exact_text || 'NOT_FOUND',
      },
    };
  });
}

function normalizeScope(scope) {
  const base = scope || {};

  const legacyTasks = Array.isArray(base.tasks) ? base.tasks : [];
  const inferredDeliverables = legacyTasks
    .map((task) => task?.task_description)
    .filter(Boolean);

  return {
    summary: base.summary || 'NOT_FOUND',
    deliverables: Array.isArray(base.deliverables) && base.deliverables.length > 0
      ? base.deliverables
      : inferredDeliverables,
    exclusions: Array.isArray(base.exclusions)
      ? base.exclusions.map((entry) => (typeof entry === 'string' ? entry : entry?.item)).filter(Boolean)
      : [],
    locations: Array.isArray(base.locations) ? base.locations : [],
    references: Array.isArray(base.references) ? base.references : [],
  };
}

export default function ResultViewer({ job }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('specs');
  const [error, setError] = useState(null);
  const jobId = job?.job_id || null;
  const jobStatus = job?.status || null;

  useEffect(() => {
    let cancelled = false;

    const loadResult = async () => {
      if (!jobId || jobStatus !== 'done') {
        setResult(null);
        setError(null);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const res = await api.get(`/jobs/${jobId}/result`);
        if (!cancelled) {
          setResult(res.data);
        }
      } catch (err) {
        if (!cancelled) {
          console.error(err);
          setError('Failed to load extracted result JSON.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadResult();
    return () => {
      cancelled = true;
    };
  }, [jobId, jobStatus]);

  const specs = useMemo(
    () => normalizeSpecs(result?.technical_specifications || []),
    [result],
  );

  const scope = useMemo(
    () => normalizeScope(result?.scope_of_work || {}),
    [result],
  );

  const highConfidenceCount = useMemo(
    () => specs.filter((s) => s.confidence >= 0.8).length,
    [specs],
  );

  if (!job) return null;

  const progress = Number(job.progress || 0);
  const startedAt = Number(job.started_at || 0);
  const now = Number(job.updated_at || job.started_at || 0);
  let etaText = '';
  if (job.status === 'running' && startedAt > 0 && now >= startedAt && progress > 0 && progress < 100) {
    const elapsed = Math.max(1, now - startedAt);
    const rate = progress / elapsed;
    if (rate > 0.05) {
      etaText = `ETA ${Math.ceil((100 - progress) / rate)}s`;
    }
  }

  if (job.status !== 'done') {
    return (
      <div className="empty-state">
        <div style={{ textAlign: 'center' }}>
          {job.status === 'error' ? (
            <>
              <XCircle size={48} className="upload-icon mx-auto mb-4" color="var(--status-red)" />
              <h3>Extraction failed</h3>
              <p className="job-message" style={{ marginTop: '10px' }}>{job.message}</p>
            </>
          ) : (
            <>
              <div style={{ marginBottom: '20px' }}>
                <span className={`badge ${job.status}`}>{job.status}</span>
              </div>
              <h3>Processing document</h3>
              <p className="job-message" style={{ marginTop: '10px', maxWidth: '640px' }}>{job.message}</p>
              <div className="progress-track" style={{ marginTop: '20px', width: '420px', height: '8px' }}>
                <div className="progress-fill" style={{ width: `${progress}%`, transition: 'width 0.5s ease' }} />
              </div>
              <div style={{ marginTop: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                {progress}% {etaText ? `- ${etaText}` : ''}
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  if (loading) return <div className="empty-state">Loading results...</div>;
  if (error) return <div className="empty-state" style={{ color: 'var(--status-red)' }}>{error}</div>;
  if (!result) return <div className="empty-state">No result payload found.</div>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="result-header">
        <div className="result-title">
          <h2>{job.filename}</h2>
          <span className="badge done" style={{ marginTop: '5px', display: 'inline-block' }}>
            Extraction Complete
          </span>
        </div>
        <div className="result-stats">
          <div className="result-stats-item">
            <strong>{specs.length}</strong> specs
          </div>
          <div className="result-stats-item">
            <strong>{highConfidenceCount}</strong> high confidence
          </div>
          <div className="result-stats-item">
            <strong>{scope.deliverables.length}</strong> deliverables
          </div>
          <div className="result-stats-item">
            <strong>{Math.round(Number(result?.accuracy_score || 0))}%</strong> accuracy
          </div>
        </div>
      </div>

      <div className="tabs">
        <div className={`tab ${activeTab === 'specs' ? 'active' : ''}`} onClick={() => setActiveTab('specs')}>
          Technical Specifications
        </div>
        <div className={`tab ${activeTab === 'scope' ? 'active' : ''}`} onClick={() => setActiveTab('scope')}>
          Scope of Work
        </div>
        <div className={`tab ${activeTab === 'qa' ? 'active' : ''}`} onClick={() => setActiveTab('qa')}>
          Ask Document
        </div>
        <div className={`tab ${activeTab === 'match' ? 'active' : ''}`} onClick={() => setActiveTab('match')}>
          Match Score
        </div>
      </div>

      {activeTab === 'specs' && <SpecsTable specs={specs} />}
      {activeTab === 'scope' && <ScopePanel scope={scope} />}
      {activeTab === 'qa' && <ChatPanel job={job} />}
      {activeTab === 'match' && <ScorePanel jobId={job.job_id} />}
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FileText, CheckCircle, XCircle } from 'lucide-react';
import SpecsTable from './SpecsTable';
import ScopePanel from './ScopePanel';

export default function ResultViewer({ job }) {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [activeTab, setActiveTab] = useState('specs');
    const [error, setError] = useState(null);

    useEffect(() => {
        if (job?.status === 'done') {
            setLoading(true);
            setError(null);
            axios.get(`/api/jobs/${job.job_id}/result`)
                .then(res => {
                    setResult(res.data);
                    setLoading(false);
                })
                .catch(err => {
                    console.error(err);
                    setError('Failed to load results');
                    setLoading(false);
                });
        } else {
            setResult(null);
        }
    }, [job?.status, job?.job_id]);

    if (job?.status !== 'done') {
        return (
            <div className="empty-state">
                <div style={{ textAlign: 'center' }}>
                    {job?.status === 'error' ? (
                        <>
                            <XCircle size={48} className="upload-icon mx-auto mb-4" color="#ef4444" />
                            <h3>Extraction Failed</h3>
                            <p className="job-message" style={{ marginTop: "10px" }}>{job.message}</p>
                        </>
                    ) : (
                        <>
                            <div style={{ marginBottom: "20px" }}>
                                <span className="badge running">Running</span>
                            </div>
                            <h3>Analyzing Document</h3>
                            <p className="job-message" style={{ marginTop: "10px" }}>{job.message}</p>
                            <div className="progress-track" style={{ marginTop: "20px", width: "300px" }}>
                                <div className="progress-fill" style={{ width: `${job.progress}%`, transition: 'width 0.5s ease' }} />
                            </div>
                        </>
                    )}
                </div>
            </div>
        );
    }

    if (loading) return <div className="empty-state">Loading results...</div>;
    if (error) return <div className="empty-state" style={{ color: 'var(--status-red)' }}>{error}</div>;
    if (!result) return null;

    const specs = result.technical_specifications || (Array.isArray(result) ? result : []);
    const scope = result.scope_of_work || { tasks: [], exclusions: [] };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div className="result-header">
                <div className="result-title">
                    <h2>{job.filename}</h2>
                    <span className="badge done" style={{ marginTop: "5px", display: "inline-block" }}>Extraction Complete</span>
                </div>
                <div className="result-stats">
                    <div className="result-stats-item">
                        <strong>{specs.length}</strong> Specs
                    </div>
                    <div className="result-stats-item">
                        <strong>{scope.tasks?.length || 0}</strong> Tasks
                    </div>
                    <div className="result-stats-item">
                        <strong>{scope.exclusions?.length || 0}</strong> Exclusions
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
            </div>

            {activeTab === 'specs' ? (
                <SpecsTable specs={specs} />
            ) : (
                <ScopePanel scope={scope} />
            )}
        </div>
    );
}

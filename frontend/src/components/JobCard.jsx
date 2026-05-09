import React from 'react';
import { Award, Clock, Loader2 } from 'lucide-react';

export default function JobCard({ job, isActive, onClick }) {
    const progress = Number(job.progress || 0);
    const matchScore = job.match_score;

    return (
        <div className={`job-card ${isActive ? 'active' : ''}`} onClick={onClick}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                <div className="job-title" style={{ margin: 0, flex: 1 }}>{job.filename}</div>
                {matchScore !== undefined && job.status === 'done' && (
                    <div className="score-badge" title={`Match Score: ${matchScore}%`}>
                        <Award size={12} />
                        {matchScore}
                    </div>
                )}
            </div>
            
            <div className="job-meta">
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span className={`badge ${job.status}`}>{job.status}</span>
                    {job.status === 'running' && (
                        <span className="running-spinner">
                            <Loader2 size={10} className="spin" />
                        </span>
                    )}
                </div>
                {job.status === 'running' && <span className="job-percentage">{progress}%</span>}
            </div>

            {(job.status === 'running' || job.status === 'queued') && (
                <div className="progress-track" style={{ height: '3px', marginTop: '10px' }}>
                    <div className="progress-fill" style={{ width: `${progress}%`, transition: 'width 0.5s ease' }} />
                </div>
            )}
            
            <div className="job-message" style={{ marginTop: '8px' }}>
                {job.status === 'running' ? job.message : (job.status === 'done' ? 'Analysis Complete' : 'Waiting in Queue')}
            </div>
        </div>
    );
}

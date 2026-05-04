import React from 'react';

export default function JobCard({ job, isActive, onClick }) {
    const progress = Number(job.progress || 0);
    const startedAt = Number(job.started_at || 0);
    const now = Number(job.updated_at || job.started_at || 0);

    let etaText = '';
    if (job.status === 'running' && startedAt > 0 && now >= startedAt && progress > 0 && progress < 100) {
        const elapsed = Math.max(1, now - startedAt);
        const rate = progress / elapsed;
        if (rate > 0.05) {
            const remain = Math.max(0, (100 - progress) / rate);
            etaText = `~${Math.ceil(remain)}s remaining`;
        }
    }

    let elapsedText = '';
    if ((job.status === 'running' || job.status === 'done') && startedAt > 0 && now >= startedAt) {
        elapsedText = `${Math.max(0, Math.floor(now - startedAt))}s elapsed`;
    }

    return (
        <div className={`job-card ${isActive ? 'active' : ''}`} onClick={onClick}>
            <div className="job-title">{job.filename}</div>
            
            <div className="job-meta">
                <span className={`badge ${job.status}`}>{job.status}</span>
                {job.status === 'running' && <span className="job-message" style={{ margin: 0 }}>{progress}%</span>}
            </div>

            {(job.status === 'running' || job.status === 'queued') && (
                <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${progress}%`, transition: 'width 0.5s ease' }} />
                </div>
            )}
            
            <div className="job-message" style={{ opacity: 0.8 }}>
                {job.status === 'running' ? job.message : (job.status === 'done' ? 'Analysis Complete' : 'Waiting in Queue')}
            </div>
            
            {job.status === 'done' && (
                <div style={{ marginTop: '12px', fontSize: '10px', color: 'var(--text-muted-soft)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {elapsedText || 'Ready'}
                </div>
            )}
        </div>
    );
}

import React from 'react';

export default function JobCard({ job, isActive, onClick }) {
    return (
        <div className={`job-card ${isActive ? 'active' : ''}`} onClick={onClick}>
            <div className="job-title">{job.filename}</div>
            <div className="job-meta">
                <span className={`badge ${job.status}`}>{job.status}</span>
                {job.progress > 0 && <span>{job.progress}%</span>}
            </div>
            {(job.status === 'running' || job.status === 'queued') && (
                <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${job.progress}%`, transition: 'width 0.5s ease' }} />
                </div>
            )}
            <div className="job-message">{job.message}</div>
        </div>
    );
}

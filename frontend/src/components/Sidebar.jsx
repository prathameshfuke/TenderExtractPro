import React from 'react';
import { FileText, Building } from 'lucide-react';
import JobCard from './JobCard';
import UploadZone from './UploadZone';

export default function Sidebar({ jobs, selectedJob, showProfile, onSelectJob, onShowProfile, onUploadComplete }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <FileText size={24} className="upload-icon" />
                <span>TenderExtractPro</span>
            </div>

            <div className="upload-zone-wrapper">
                <UploadZone onUploadComplete={onUploadComplete} />
                <button 
                    onClick={onShowProfile}
                    className={`profile-button ${showProfile ? 'active' : ''}`}
                    style={{
                        width: '100%',
                        marginTop: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '12px',
                        background: showProfile ? 'var(--primary-dark)' : 'var(--surface-light)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text)',
                        cursor: 'pointer',
                        fontWeight: 500,
                        transition: 'background 0.2s'
                    }}
                >
                    <Building size={18} />
                    Company Profile
                </button>
            </div>

            <div className="job-list">
                {jobs.length === 0 && (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '12px' }}>
                        No extraction jobs yet.
                    </div>
                )}

                {jobs.map(job => (
                    <JobCard
                        key={job.job_id}
                        job={job}
                        isActive={selectedJob?.job_id === job.job_id}
                        onClick={() => onSelectJob(job)}
                    />
                ))}
            </div>
        </aside>
    );
}

import React from 'react';
import { FileText, Building } from 'lucide-react';
import JobCard from './JobCard';
import UploadZone from './UploadZone';

export default function Sidebar({ jobs, selectedJob, showProfile, onSelectJob, onShowProfile, onUploadComplete }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <span className="tagline">Intelligence Platform</span>
                <h1>TenderExtractPro</h1>
            </div>

            <div className="upload-zone-wrapper">
                <UploadZone onUploadComplete={onUploadComplete} />
                <button 
                    onClick={onShowProfile}
                    className={`outline-pill ${showProfile ? 'active' : ''}`}
                    style={{
                        width: '100%',
                        marginTop: '16px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '12px',
                    }}
                >
                    <Building size={16} />
                    Company Profile
                </button>
            </div>

            <div className="job-list">
                {jobs.map(job => (
                    <JobCard
                        key={job.job_id}
                        job={job}
                        isActive={selectedJob?.job_id === job.job_id && !showProfile}
                        onClick={() => onSelectJob(job)}
                    />
                ))}
            </div>
        </aside>
    );
}

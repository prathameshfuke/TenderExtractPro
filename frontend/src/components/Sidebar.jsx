import React, { useState } from 'react';
import { FileText, Building, ArrowUpDown, Filter, User } from 'lucide-react';
import JobCard from './JobCard';
import UploadZone from './UploadZone';

export default function Sidebar({ 
  jobs, 
  selectedJob, 
  showProfile, 
  onSelectJob, 
  onShowProfile, 
  onUploadComplete,
  sortBy,
  onSortChange
}) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="tagline">Intelligence Platform</span>
                    <button 
                        className={`icon-btn ${showProfile ? 'active' : ''}`}
                        onClick={onShowProfile}
                        title="Company Profile"
                    >
                        <User size={18} />
                    </button>
                </div>
                <h1>TenderExtractPro</h1>
            </div>

            <div className="sidebar-controls" style={{ padding: '0 24px 16px', borderBottom: '1px solid var(--hairline)' }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button 
                        className={`sort-pill ${sortBy === 'recency' ? 'active' : ''}`}
                        onClick={() => onSortChange('recency')}
                    >
                        <ArrowUpDown size={12} />
                        Recent
                    </button>
                    <button 
                        className={`sort-pill ${sortBy === 'score' ? 'active' : ''}`}
                        onClick={() => onSortChange('score')}
                    >
                        <Filter size={12} />
                        Best Match
                    </button>
                </div>
            </div>

            <div className="upload-zone-wrapper">
                <UploadZone onUploadComplete={onUploadComplete} />
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

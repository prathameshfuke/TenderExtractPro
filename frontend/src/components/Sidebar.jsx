import React from 'react';
import { FileText } from 'lucide-react';
import JobCard from './JobCard';
import UploadZone from './UploadZone';

export default function Sidebar({ jobs, selectedJob, onSelectJob, onUploadComplete }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <FileText size={24} className="upload-icon" />
                <span>TenderExtractPro</span>
            </div>

            <div className="upload-zone-wrapper">
                <UploadZone onUploadComplete={onUploadComplete} />
            </div>

            <div className="job-list">
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

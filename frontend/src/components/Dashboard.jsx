import React from 'react';
import { ArrowUpDown, Filter, Search, FileText, Clock, Award, ChevronRight } from 'lucide-react';

export default function Dashboard({ jobs, onSelectJob, sortBy, onSortChange }) {
  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div>
          <h1>Your Workspace</h1>
          <p>Manage and analyze your tender documents</p>
        </div>
        
        <div className="dashboard-controls">
          <div className="dashboard-search">
            <Search size={18} />
            <input type="text" placeholder="Filter documents by name or content..." />
          </div>
          
          <div className="dashboard-filters">
            <span className="filter-label">Sort by</span>
            <div className="filter-toggle">
              <button 
                className={sortBy === 'recency' ? 'active' : ''}
                onClick={() => onSortChange('recency')}
              >
                Date
              </button>
              <button 
                className={sortBy === 'score' ? 'active' : ''}
                onClick={() => onSortChange('score')}
              >
                Match
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="jobs-grid">
        {jobs.length === 0 ? (
          <div className="empty-grid-state">
            <div className="empty-icon">
              <FileText size={48} />
            </div>
            <h3>No tenders analyzed yet</h3>
            <p>Upload a PDF document to start the extraction pipeline.</p>
          </div>
        ) : (
          jobs.map((job, idx) => (
            <div 
              key={job.job_id} 
              className={`dashboard-card ${job.status} animate-fade`}
              style={{ animationDelay: `${idx * 0.05}s` }}
              onClick={() => onSelectJob(job)}
            >
              <div className="card-header">
                <div className="card-type-icon">
                  <FileText size={20} />
                </div>
                <div className={`status-badge ${job.status}`}>
                  {job.status}
                </div>
              </div>
              
              <div className="card-body">
                <h3 className="card-title">{job.filename}</h3>
                <p className="card-msg">{job.message}</p>
                
                {job.status === 'running' && (
                  <div className="card-progress">
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${job.progress}%` }}></div>
                    </div>
                    <span className="progress-text">{job.progress}%</span>
                  </div>
                )}
              </div>

              <div className="card-footer">
                {job.match_score !== undefined && job.status === 'done' ? (
                  <div className="card-score">
                    <Award size={14} />
                    <span>{job.match_score}% Match</span>
                  </div>
                ) : (
                  <div className="card-date">
                     {new Date(job.created_at * 1000).toLocaleDateString()}
                  </div>
                )}
                <div className="card-action">
                  <span>View Details</span>
                  <ChevronRight size={14} />
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

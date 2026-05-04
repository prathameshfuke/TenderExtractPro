import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { FileText } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ResultViewer from './components/ResultViewer';
import ProfilePanel from './components/ProfilePanel';

const api = axios.create({ baseURL: '/api' });

function sortJobsByRecency(items) {
  return [...items].sort((a, b) => {
    const ta = Number(a?.created_at || 0);
    const tb = Number(b?.created_at || 0);
    return tb - ta;
  });
}

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);
  const [showProfile, setShowProfile] = useState(false);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) || null,
    [jobs, selectedJobId],
  );

  useEffect(() => {
    let isMounted = true;

    const loadJobs = async () => {
      try {
        const res = await api.get('/jobs');
        if (!isMounted) return;
        setBackendOnline(true);

        const sorted = sortJobsByRecency(Array.isArray(res.data) ? res.data : []);
        setJobs(sorted);
        if (!selectedJobId && sorted.length > 0) {
          setSelectedJobId(sorted[0].job_id);
        }
      } catch (err) {
        setBackendOnline(false);
        console.error('Failed to load jobs', err);
      }
    };

    loadJobs();
    return () => {
      isMounted = false;
    };
  }, [selectedJobId]);

  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await api.get('/jobs');
        const incoming = Array.isArray(res.data) ? res.data : [];
        setBackendOnline(true);
        setJobs(sortJobsByRecency(incoming));
      } catch (err) {
        setBackendOnline(false);
        console.error('Failed to refresh jobs', err);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const handleUploadComplete = (newJob) => {
    setJobs((prev) => sortJobsByRecency([newJob, ...prev]));
    setSelectedJobId(newJob.job_id);
    setShowProfile(false);
  };

  return (
    <div className="app-container">
      {/* Atmospheric Orbs */}
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>
      <div className="orb orb-4"></div>

      {!backendOnline && (
        <div style={{
          position: 'fixed',
          top: 12,
          right: 12,
          zIndex: 999,
          background: 'var(--error)',
          color: 'white',
          padding: '10px 16px',
          borderRadius: '9999px',
          fontSize: '0.85rem',
          fontWeight: 500,
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}>
          API Offline
        </div>
      )}

      <Sidebar
        jobs={jobs}
        selectedJob={selectedJob}
        showProfile={showProfile}
        onSelectJob={(job) => { setSelectedJobId(job.job_id); setShowProfile(false); }}
        onShowProfile={() => setShowProfile(true)}
        onUploadComplete={handleUploadComplete}
      />

      <main className="main-content">
        {showProfile ? (
          <ProfilePanel />
        ) : !selectedJob ? (
          <div className="empty-state">
            <h1>TenderExtractPro</h1>
            <p>
              An editorial approach to tender extraction. 
              Upload a document to begin the deep-analysis pipeline.
            </p>
            <div style={{ marginTop: '32px' }}>
              <button 
                className="outline-pill"
                onClick={() => document.getElementById('sidebar-upload-trigger')?.click()}
              >
                Get Started
              </button>
            </div>
          </div>
        ) : (
          <ResultViewer job={selectedJob} />
        )}
      </main>
    </div>
  );
}

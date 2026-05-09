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
  const [sortBy, setSortBy] = useState('recency'); // 'recency' or 'score'

  const processedJobs = useMemo(() => {
    let sorted = [...jobs];
    if (sortBy === 'recency') {
      sorted.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    } else if (sortBy === 'score') {
      sorted.sort((a, b) => (b.match_score || 0) - (a.match_score || 0));
    }
    return sorted;
  }, [jobs, sortBy]);

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
        const incoming = Array.isArray(res.data) ? res.data : [];
        setJobs(incoming);
        
        if (!selectedJobId && incoming.length > 0) {
          // If sorting by score, we might want the highest score first
          const firstJob = sortBy === 'score' 
            ? [...incoming].sort((a,b) => (b.match_score || 0) - (a.match_score || 0))[0]
            : [...incoming].sort((a,b) => (b.created_at || 0) - (a.created_at || 0))[0];
          setSelectedJobId(firstJob.job_id);
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
  }, []);

  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await api.get('/jobs');
        const incoming = Array.isArray(res.data) ? res.data : [];
        setBackendOnline(true);
        setJobs(incoming);
      } catch (err) {
        setBackendOnline(false);
        console.error('Failed to refresh jobs', err);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const handleUploadComplete = (newJob) => {
    setJobs((prev) => [newJob, ...prev]);
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
        jobs={processedJobs}
        selectedJob={selectedJob}
        showProfile={showProfile}
        onSelectJob={(job) => { setSelectedJobId(job.job_id); setShowProfile(false); }}
        onShowProfile={() => setShowProfile(true)}
        onUploadComplete={handleUploadComplete}
        sortBy={sortBy}
        onSortChange={setSortBy}
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

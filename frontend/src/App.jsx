import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { FileText, X } from 'lucide-react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ResultViewer from './components/ResultViewer';
import ProfilePanel from './components/ProfilePanel';
import UploadZone from './components/UploadZone';

const api = axios.create({ baseURL: '/api' });

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);
  const [showProfile, setShowProfile] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
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
    setShowUploadModal(false);
  };

  const handleLogoClick = () => {
    setSelectedJobId(null);
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
          zIndex: 9999,
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

      <Header 
        onShowProfile={() => { setShowProfile(true); setSelectedJobId(null); }}
        onUploadClick={() => setShowUploadModal(true)}
        onLogoClick={handleLogoClick}
        showProfile={showProfile}
        hasSelectedJob={!!selectedJobId}
      />

      <main className="main-content">
        {showProfile ? (
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <ProfilePanel />
          </div>
        ) : !selectedJobId ? (
          <Dashboard 
            jobs={processedJobs}
            onSelectJob={(job) => setSelectedJobId(job.job_id)}
            sortBy={sortBy}
            onSortChange={setSortBy}
          />
        ) : (
          <ResultViewer job={selectedJob} onBack={handleLogoClick} />
        )}
      </main>

      {showUploadModal && (
        <div className="modal-overlay" onClick={() => setShowUploadModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowUploadModal(false)}>
              <X size={20} />
            </button>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <h2 style={{ fontSize: '24px', fontWeight: 300, marginBottom: '8px' }}>Analyze New Tender</h2>
              <p style={{ color: 'var(--text-muted)' }}>Upload a PDF to begin the analysis</p>
            </div>
            <UploadZone onUploadComplete={handleUploadComplete} />
          </div>
        </div>
      )}
    </div>
  );
}

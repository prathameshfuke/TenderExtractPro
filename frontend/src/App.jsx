import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { X } from 'lucide-react';

import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ResultViewer from './components/ResultViewer';
import ProfilePanel from './components/ProfilePanel';
import UploadZone from './components/UploadZone';
import LandingPage from './components/LandingPage';
import Documentation from './components/Documentation';

const api = axios.create({ baseURL: '/api' });

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);
  const [showProfile, setShowProfile] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [sortBy, setSortBy] = useState('recency');
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => (prev === 'light' ? 'dark' : 'light'));

  const processedJobs = useMemo(() => {
    if (!Array.isArray(jobs)) return [];
    let sorted = [...jobs];
    if (sortBy === 'recency') {
      sorted.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    } else if (sortBy === 'score') {
      sorted.sort((a, b) => (b.match_score || 0) - (a.match_score || 0));
    }
    return sorted;
  }, [jobs, sortBy]);

  const selectedJob = useMemo(
    () => jobs.find((j) => j.job_id === selectedJobId) || null,
    [jobs, selectedJobId]
  );

  useEffect(() => {
    const loadJobs = async () => {
      try {
        const res = await api.get('/jobs');
        setBackendOnline(true);
        setJobs(Array.isArray(res.data) ? res.data : []);
      } catch (err) {
        setBackendOnline(false);
      }
    };
    loadJobs();
    const timer = setInterval(loadJobs, 4000);
    return () => clearInterval(timer);
  }, []);

  const handleUploadComplete = (newJob) => {
    setJobs((prev) => [newJob, ...prev]);
    setSelectedJobId(newJob.job_id);
    setShowProfile(false);
    setShowUploadModal(false);
    setHasStarted(true);
  };

  const handleLogoClick = () => {
    setSelectedJobId(null);
    setShowProfile(false);
    setShowDocs(false);
  };

  const isLanding = !hasStarted && !showDocs && jobs.length === 0;

  return (
    <div className="app-container">
      {/* Background Ambience */}
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>
      <div className="orb orb-4"></div>

      {!backendOnline && (
        <div style={{
          position: 'fixed', top: 12, right: 12, zIndex: 9999,
          background: 'var(--error)', color: 'white', padding: '10px 16px',
          borderRadius: '9999px', fontSize: '0.85rem', fontWeight: 600,
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}>
          API Offline
        </div>
      )}

      {!isLanding && (
        <Header 
          onShowProfile={() => { setShowProfile(true); setSelectedJobId(null); setHasStarted(true); setShowDocs(false); }}
          onUploadClick={() => setShowUploadModal(true)}
          onLogoClick={handleLogoClick}
          showProfile={showProfile}
          hasSelectedJob={!!selectedJobId}
          theme={theme}
          onToggleTheme={toggleTheme}
          onShowDocs={() => { setShowDocs(true); setSelectedJobId(null); setShowProfile(false); setHasStarted(true); }}
          showDocs={showDocs}
        />
      )}

      <main className="main-content">
        {isLanding ? (
          <LandingPage 
            onGetStarted={() => setHasStarted(true)} 
            onShowDocs={() => setShowDocs(true)}
          />
        ) : showDocs ? (
          <Documentation onBack={() => setShowDocs(false)} />
        ) : showProfile ? (
          <ProfilePanel />
        ) : (!selectedJobId || (selectedJobId && !selectedJob)) ? (
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
              <h2 style={{ fontSize: '24px', fontWeight: 800, marginBottom: '8px', letterSpacing: '-0.02em', color: 'var(--text-ink)' }}>Analyze New Tender</h2>
              <p style={{ color: 'var(--text-muted)' }}>Upload a PDF to begin the analysis</p>
            </div>
            <UploadZone onUploadComplete={handleUploadComplete} />
          </div>
        </div>
      )}
    </div>
  );
}

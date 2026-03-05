import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { FileText } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ResultViewer from './components/ResultViewer';

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

        const sorted = sortJobsByRecency(Array.isArray(res.data) ? res.data : []);
        setJobs(sorted);
        if (!selectedJobId && sorted.length > 0) {
          setSelectedJobId(sorted[0].job_id);
        }
      } catch (err) {
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
        setJobs(sortJobsByRecency(incoming));
      } catch (err) {
        console.error('Failed to refresh jobs', err);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const handleUploadComplete = (newJob) => {
    setJobs((prev) => sortJobsByRecency([newJob, ...prev]));
    setSelectedJobId(newJob.job_id);
  };

  return (
    <div className="app-container">
      <Sidebar
        jobs={jobs}
        selectedJob={selectedJob}
        onSelectJob={(job) => setSelectedJobId(job.job_id)}
        onUploadComplete={handleUploadComplete}
      />

      <main className="main-content">
        {!selectedJob ? (
          <div className="empty-state">
            <div style={{ textAlign: 'center' }}>
              <FileText size={48} className="upload-icon" style={{ opacity: 0.5, marginBottom: '12px' }} />
              <h3 style={{ marginBottom: '8px' }}>Upload a tender PDF to start extraction</h3>
              <p style={{ color: 'var(--text-muted)' }}>
                The pipeline will process specs, scope, and grounded source evidence.
              </p>
            </div>
          </div>
        ) : (
          <ResultViewer job={selectedJob} />
        )}
      </main>
    </div>
  );
}

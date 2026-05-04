import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Save, Building } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

export default function ProfilePanel() {
  const [profileText, setProfileText] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await api.get('/profile');
        setProfileText(JSON.stringify(res.data, null, 2));
      } catch (err) {
        console.error(err);
      }
    };
    fetchProfile();
  }, []);

  const handleSave = async () => {
    try {
      setSaving(true);
      setMessage("");
      const parsed = JSON.parse(profileText);
      await api.post('/profile', parsed);
      setMessage("Profile saved successfully.");
    } catch (err) {
      setMessage("Invalid JSON or failed to save.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="panel" style={{ padding: '32px', maxWidth: '800px', margin: '0 auto', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <Building size={28} color="var(--primary)" />
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600, color: 'var(--text)' }}>
          Company Profile Settings
        </h2>
      </div>

      <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
        Define your company's capabilities, budget limits, and exclusions here in JSON format. 
        This profile will be used by the LLM to score and rank incoming tenders.
      </p>

      <textarea
        style={{
          width: '100%',
          height: '400px',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          color: 'var(--text)',
          fontFamily: 'var(--font-mono)',
          padding: '16px',
          fontSize: '0.9rem',
          marginBottom: '24px',
          resize: 'vertical'
        }}
        value={profileText}
        onChange={(e) => setProfileText(e.target.value)}
      />

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <button 
          onClick={handleSave} 
          disabled={saving}
          style={{
            background: 'var(--primary)',
            color: 'white',
            border: 'none',
            padding: '10px 24px',
            borderRadius: '6px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: 500
          }}
        >
          <Save size={18} />
          {saving ? 'Saving...' : 'Save Profile'}
        </button>
        {message && <span style={{ color: message.includes("success") ? 'var(--status-green)' : 'var(--status-red)' }}>{message}</span>}
      </div>
    </div>
  );
}

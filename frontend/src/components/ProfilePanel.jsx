import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Save, Building, X, Plus } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

function TagInput({ tags, setTags, placeholder }) {
  const [input, setInput] = useState('');

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const val = input.trim();
      if (val && !tags.includes(val)) {
        setTags([...tags, val]);
        setInput('');
      }
    }
  };

  const removeTag = (tagToRemove) => {
    setTags(tags.filter(t => t !== tagToRemove));
  };

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
        {tags.map((tag, idx) => (
          <div key={idx} style={{ 
            background: 'var(--surface-light)', 
            padding: '4px 10px', 
            borderRadius: '16px', 
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            border: '1px solid var(--border)'
          }}>
            {tag}
            <button 
              type="button" 
              onClick={() => removeTag(tag)}
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 0, display: 'flex' }}
            >
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '8px' }}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          style={{
            flex: 1, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '6px',
            color: 'var(--text)', padding: '8px 12px', fontSize: '0.9rem'
          }}
        />
        <button 
          type="button"
          onClick={() => handleKeyDown({ key: 'Enter', preventDefault: () => {} })}
          style={{ background: 'var(--surface-light)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '6px', padding: '0 12px', cursor: 'pointer' }}
        >
          <Plus size={16} />
        </button>
      </div>
    </div>
  );
}

export default function ProfilePanel() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const [companyName, setCompanyName] = useState("");
  const [budget, setBudget] = useState(0);
  const [capabilities, setCapabilities] = useState([]);
  const [certifications, setCertifications] = useState([]);
  const [locations, setLocations] = useState([]);
  const [exclusions, setExclusions] = useState([]);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await api.get('/profile');
        const data = res.data || {};
        setCompanyName(data.company_name || "");
        setBudget(data.max_project_budget_usd || 0);
        setCapabilities(data.core_capabilities || []);
        setCertifications(data.certifications || []);
        setLocations(data.preferred_locations || []);
        setExclusions(data.exclusions || []);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSave = async () => {
    try {
      setSaving(true);
      setMessage("");
      const payload = {
        company_name: companyName,
        core_capabilities: capabilities,
        certifications: certifications,
        max_project_budget_usd: Number(budget),
        preferred_locations: locations,
        exclusions: exclusions
      };
      await api.post('/profile', payload);
      setMessage("Profile saved successfully.");
    } catch (err) {
      setMessage("Failed to save profile.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="empty-state">Loading profile...</div>;

  return (
    <div className="panel" style={{ padding: '64px', maxWidth: '1000px', margin: '0 auto', height: '100%', overflowY: 'auto' }}>
      <div style={{ marginBottom: '48px' }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 300, fontSize: '42px', color: 'var(--text-ink)', marginBottom: '16px' }}>
          Company Profile
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '18px', lineHeight: 1.6, maxWidth: '600px' }}>
          Define your operational parameters. This data grounds our evaluation engine, 
          allowing it to rank tenders against your core strengths.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '48px', marginBottom: '64px' }}>
        <div style={{ gridColumn: 'span 2' }}>
           <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '12px' }}>
            Official Entity Name
          </label>
          <input 
            type="text" 
            value={companyName} 
            onChange={(e) => setCompanyName(e.target.value)}
            style={{ width: '100%', background: 'var(--surface-card)', border: '1px solid var(--hairline)', borderRadius: '12px', color: 'var(--text-ink)', padding: '16px 20px', fontSize: '16px', outline: 'none' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '12px' }}>
            Maximum Project Valuation (USD)
          </label>
          <input 
            type="number" 
            value={budget} 
            onChange={(e) => setBudget(e.target.value)}
            style={{ width: '100%', background: 'var(--surface-card)', border: '1px solid var(--hairline)', borderRadius: '12px', color: 'var(--text-ink)', padding: '16px 20px', fontSize: '16px', outline: 'none' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '12px' }}>
            Core Capabilities
          </label>
          <TagInput tags={capabilities} setTags={setCapabilities} placeholder="e.g. EPC Contracting" />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '12px' }}>
            Strategic Certifications
          </label>
          <TagInput tags={certifications} setTags={setCertifications} placeholder="e.g. ASME VIII" />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '12px' }}>
            Preferred Jurisdictions
          </label>
          <TagInput tags={locations} setTags={setLocations} placeholder="e.g. SE Asia" />
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '24px', paddingTop: '32px', borderTop: '1px solid var(--hairline)' }}>
        <button 
          onClick={handleSave} 
          disabled={saving}
          className="primary-pill"
          style={{ display: 'flex', alignItems: 'center', gap: '12px' }}
        >
          <Save size={18} />
          {saving ? 'Synchronizing...' : 'Save Profile'}
        </button>
        {message && <span style={{ fontSize: '15px', color: message.includes("success") ? 'var(--success)' : 'var(--error)' }}>{message}</span>}
      </div>
    </div>
  );
}

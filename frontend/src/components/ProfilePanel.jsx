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
    <div className="panel" style={{ padding: '32px', maxWidth: '800px', margin: '0 auto', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <Building size={28} color="var(--primary)" />
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600, color: 'var(--text)' }}>
          Company Profile Dashboard
        </h2>
      </div>

      <p style={{ color: 'var(--text-muted)', marginBottom: '32px', lineHeight: '1.5' }}>
        Configure your company details below. This data is used by the LLM to score and rank incoming tenders.
      </p>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
          Company Name
        </label>
        <input 
          type="text" 
          value={companyName} 
          onChange={(e) => setCompanyName(e.target.value)}
          style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '6px', color: 'var(--text)', padding: '10px 12px', fontSize: '1rem' }}
        />
      </div>

      <div style={{ marginBottom: '24px' }}>
        <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
          Max Project Budget (USD)
        </label>
        <input 
          type="number" 
          value={budget} 
          onChange={(e) => setBudget(e.target.value)}
          style={{ width: '100%', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '6px', color: 'var(--text)', padding: '10px 12px', fontSize: '1rem' }}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
            Core Capabilities
          </label>
          <TagInput tags={capabilities} setTags={setCapabilities} placeholder="e.g. HVAC Installation" />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
            Certifications
          </label>
          <TagInput tags={certifications} setTags={setCertifications} placeholder="e.g. ISO 9001" />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
            Preferred Locations
          </label>
          <TagInput tags={locations} setTags={setLocations} placeholder="e.g. North America" />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
            Exclusions
          </label>
          <TagInput tags={exclusions} setTags={setExclusions} placeholder="e.g. Nuclear Facilities" />
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
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

import React, { useMemo, useState } from 'react';
import { Search, Filter } from 'lucide-react';
import SourceDrawer from './SourceDrawer';

function confidenceBucket(score) {
  if (score >= 0.8) return 'high';
  if (score >= 0.5) return 'medium';
  return 'low';
}

function specsSummary(specMap) {
  const entries = Object.entries(specMap || {});
  if (entries.length === 0) return 'No explicit parameter map';
  return entries
    .slice(0, 3)
    .map(([k, v]) => `${k}: ${v}`)
    .join(' | ');
}

export default function SpecsTable({ specs }) {
  const [filterText, setFilterText] = useState('');
  const [confidenceFilter, setConfidenceFilter] = useState('all');
  const [selectedSpec, setSelectedSpec] = useState(null);

  const filteredSpecs = useMemo(() => {
    return (Array.isArray(specs) ? specs : []).filter((spec) => {
      const bucket = confidenceBucket(Number(spec.confidence || 0));
      if (confidenceFilter !== 'all' && bucket !== confidenceFilter) return false;

      if (!filterText.trim()) return true;
      const search = filterText.toLowerCase();
      const blob = `${spec.component} ${specsSummary(spec.specs)} ${spec.source?.exact_text || ''}`.toLowerCase();
      return blob.includes(search);
    });
  }, [specs, filterText, confidenceFilter]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div className="controls-bar">
        <div className="input-group">
          <Search size={18} className="input-icon" />
          <input
            type="text"
            className="text-input"
            placeholder="Search component, parameter, value, or source text..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
        </div>

        <div className="input-group" style={{ background: 'transparent', border: 'none' }}>
          <Filter size={18} className="input-icon" />
          <select
            className="select-input"
            value={confidenceFilter}
            onChange={(e) => setConfidenceFilter(e.target.value)}
          >
            <option value="all">All confidence</option>
            <option value="high">High (&gt;= 0.80)</option>
            <option value="medium">Medium (0.50 - 0.79)</option>
            <option value="low">Low (&lt; 0.50)</option>
          </select>
        </div>

        <div style={{ marginLeft: 'auto', alignSelf: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Showing {filteredSpecs.length} of {specs.length} specs
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Spec Summary</th>
              <th>Confidence</th>
              <th>Page</th>
              <th>Clause</th>
            </tr>
          </thead>
          <tbody>
            {filteredSpecs.map((spec, i) => {
              const bucket = confidenceBucket(Number(spec.confidence || 0));
              return (
                <tr key={`${spec.component}-${i}`} onClick={() => setSelectedSpec(spec)}>
                  <td style={{ fontWeight: 500, color: 'var(--text-main)' }}>{spec.component}</td>
                  <td style={{ maxWidth: '460px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {specsSummary(spec.specs)}
                  </td>
                  <td>
                    <span className={`badge ${bucket}`}>
                      {(Number(spec.confidence || 0)).toFixed(2)}
                    </span>
                  </td>
                  <td>{spec.source?.page || '-'}</td>
                  <td>{spec.source?.clause && spec.source.clause !== 'NOT_FOUND' ? spec.source.clause : '-'}</td>
                </tr>
              );
            })}

            {filteredSpecs.length === 0 && (
              <tr>
                <td colSpan="5" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                  No specifications match the active filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <SourceDrawer
        isOpen={!!selectedSpec}
        onClose={() => setSelectedSpec(null)}
        source={selectedSpec?.source}
        title={selectedSpec?.component || 'Source Reference'}
        specs={selectedSpec?.specs || {}}
        confidence={selectedSpec?.confidence || 0}
      />
    </div>
  );
}

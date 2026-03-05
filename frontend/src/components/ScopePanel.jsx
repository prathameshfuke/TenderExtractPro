import React from 'react';

function ListCard({ title, items, emptyText }) {
  return (
    <section className="scope-section">
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p style={{ color: 'var(--text-muted)' }}>{emptyText}</p>
      ) : (
        <div style={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-color)' }}>
          {items.map((item, i) => (
            <div key={`${title}-${i}`} className="exclusion-item">
              {item}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function ScopePanel({ scope }) {
  const summary = scope?.summary || 'NOT_FOUND';
  const deliverables = Array.isArray(scope?.deliverables) ? scope.deliverables : [];
  const exclusions = Array.isArray(scope?.exclusions) ? scope.exclusions : [];
  const locations = Array.isArray(scope?.locations) ? scope.locations : [];
  const references = Array.isArray(scope?.references) ? scope.references : [];

  return (
    <div className="scope-panel">
      <section className="scope-section">
        <h3>Summary</h3>
        <div className="task-card">
          <div className="task-desc" style={{ fontSize: '1rem', lineHeight: 1.6 }}>
            {summary !== 'NOT_FOUND' ? summary : 'No scope summary was extracted.'}
          </div>
        </div>
      </section>

      <ListCard
        title={`Deliverables (${deliverables.length})`}
        items={deliverables}
        emptyText="No explicit deliverables identified."
      />

      <ListCard
        title={`Exclusions (${exclusions.length})`}
        items={exclusions}
        emptyText="No exclusions were explicitly identified."
      />

      <ListCard
        title={`Locations (${locations.length})`}
        items={locations}
        emptyText="No project location details found."
      />

      <ListCard
        title={`References (${references.length})`}
        items={references}
        emptyText="No clause references extracted for scope."
      />
    </div>
  );
}

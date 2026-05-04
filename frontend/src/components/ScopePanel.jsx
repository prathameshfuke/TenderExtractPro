import React from 'react';

function ListCard({ title, items, emptyText }) {
  return (
    <section className="scope-section" style={{ marginBottom: '48px' }}>
      <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 300, fontSize: '24px', color: 'var(--text-ink)', marginBottom: '20px' }}>{title}</h3>
      {items.length === 0 ? (
        <p style={{ color: 'var(--text-muted-soft)', fontSize: '15px' }}>{emptyText}</p>
      ) : (
        <div style={{ backgroundColor: 'var(--surface-card)', border: '1px solid var(--hairline)', borderRadius: '16px', overflow: 'hidden' }}>
          {items.map((item, i) => (
            <div key={`${title}-${i}`} className="exclusion-item" style={{ padding: '20px 24px', borderBottom: i === items.length - 1 ? 'none' : '1px solid var(--hairline)', fontSize: '15px', display: 'flex', gap: '16px' }}>
              <span style={{ color: 'var(--text-muted-soft)' }}>{String(i + 1).padStart(2, '0')}</span>
              <span>{item}</span>
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
    <div className="scope-panel" style={{ padding: '48px' }}>
      <section className="scope-section" style={{ marginBottom: '64px' }}>
        <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 300, fontSize: '24px', color: 'var(--text-ink)', marginBottom: '20px' }}>Summary</h3>
        <div style={{ fontSize: '20px', lineHeight: 1.5, color: 'var(--text-body)', fontWeight: 400, maxWidth: '800px' }}>
          {summary !== 'NOT_FOUND' ? summary : 'No scope summary was extracted.'}
        </div>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '48px' }}>
        <ListCard
          title="Deliverables"
          items={deliverables}
          emptyText="No explicit deliverables identified."
        />

        <ListCard
          title="Exclusions"
          items={exclusions}
          emptyText="No exclusions were explicitly identified."
        />

        <ListCard
          title="Project Locations"
          items={locations}
          emptyText="No project location details found."
        />

        <ListCard
          title="Clause References"
          items={references}
          emptyText="No clause references extracted for scope."
        />
      </div>
    </div>
  );
}

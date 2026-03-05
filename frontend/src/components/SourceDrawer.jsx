import React from 'react';
import { X, FileText } from 'lucide-react';

export default function SourceDrawer({ isOpen, onClose, source, title, specs = {}, confidence = 0 }) {
    if (!isOpen) return <div className="source-drawer"></div>;

    const entries = Object.entries(specs || {});
    const confidenceValue = Number(confidence || 0).toFixed(2);

    return (
        <div className={`source-drawer ${isOpen ? 'open' : ''}`}>
            <div className="drawer-header">
                <h3>{title}</h3>
                <button className="close-btn" onClick={onClose}><X size={20} /></button>
            </div>

            <div className="drawer-content">
                <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '10px' }}>
                    Evidence Match
                </h4>

                {source ? (
                    <>
                        <div className="source-meta">
                            <span><FileText size={14} style={{ display: 'inline', marginBottom: '-2px', marginRight: '5px' }} /> Source Text</span>
                            <span className="badge queued">Page {source.page || '-'}</span>
                        </div>
                        <div className="source-text-box">
                            {source.exact_text || "No exact text snippet available."}
                        </div>

                        <div style={{ marginTop: '16px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                            Clause: {source.clause && source.clause !== 'NOT_FOUND' ? source.clause : '-'}
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '4px' }}>
                            Confidence: {confidenceValue}
                        </div>

                        <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginTop: '20px', marginBottom: '10px' }}>
                            Parameter Map
                        </h4>
                        {entries.length === 0 ? (
                            <div className="source-text-box" style={{ marginTop: 0, color: 'var(--text-muted)' }}>
                                No structured parameters were extracted for this component.
                            </div>
                        ) : (
                            <div style={{ border: '1px solid var(--border-color)' }}>
                                {entries.map(([key, value]) => (
                                    <div key={key} style={{ display: 'flex', borderBottom: '1px solid var(--border-color)' }}>
                                        <div style={{ width: '38%', padding: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                                            {key}
                                        </div>
                                        <div style={{ width: '62%', padding: '10px', fontSize: '0.82rem' }}>
                                            {value}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {source.chunk_id && (
                            <div style={{ marginTop: '20px', fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                Ref ID: {source.chunk_id}
                            </div>
                        )}
                    </>
                ) : (
                    <div className="source-text-box" style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        Source tracing metadata is missing for this extracted item.
                    </div>
                )}
            </div>
        </div>
    );
}

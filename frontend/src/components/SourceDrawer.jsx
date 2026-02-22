import React from 'react';
import { X, FileText } from 'lucide-react';

export default function SourceDrawer({ isOpen, onClose, source, title }) {
    if (!isOpen) return <div className="source-drawer"></div>;

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
                            {source.page && <span className="badge queued">Page {source.page}</span>}
                        </div>
                        <div className="source-text-box">
                            {source.exact_text || "No exact text snippet available."}
                        </div>

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

import React from 'react';

export default function ScopePanel({ scope }) {
    const { tasks = [], exclusions = [] } = scope;

    return (
        <div className="scope-panel">
            <div className="scope-section">
                <h3>Project Tasks & Deliverables</h3>
                {tasks.length === 0 ? (
                    <p style={{ color: 'var(--text-muted)' }}>No explicit tasks identified in the document.</p>
                ) : (
                    tasks.map((task, i) => (
                        <div key={i} className="task-card">
                            <div className="task-desc">{task.task_description}</div>
                            <div className="task-grid">
                                {task.timeline && task.timeline !== 'NOT_FOUND' && (
                                    <>
                                        <div className="task-grid-label">Timeline</div>
                                        <div>{task.timeline}</div>
                                    </>
                                )}

                                {task.deliverables && task.deliverables.length > 0 && task.deliverables[0] !== 'NOT_FOUND' && (
                                    <>
                                        <div className="task-grid-label">Deliverables</div>
                                        <div>
                                            <ul>
                                                {task.deliverables.map((d, j) => <li key={j}>{d}</li>)}
                                            </ul>
                                        </div>
                                    </>
                                )}

                                {task.dependencies && task.dependencies.length > 0 && task.dependencies[0] !== 'NOT_FOUND' && (
                                    <>
                                        <div className="task-grid-label">Dependencies</div>
                                        <div>
                                            <ul>
                                                {task.dependencies.map((d, j) => <li key={j}>{d}</li>)}
                                            </ul>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            <div className="scope-section">
                <h3>Exclusions</h3>
                {exclusions.length === 0 ? (
                    <p style={{ color: 'var(--text-muted)' }}>No explicit exclusions identified.</p>
                ) : (
                    <div style={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-color)' }}>
                        {exclusions.map((excl, i) => (
                            <div key={i} className="exclusion-item">
                                {excl.item || excl}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

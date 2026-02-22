import React, { useState } from 'react';
import { Search, Filter } from 'lucide-react';
import SourceDrawer from './SourceDrawer';

export default function SpecsTable({ specs }) {
    const [filterText, setFilterText] = useState('');
    const [confidenceFilter, setConfidenceFilter] = useState('ALL');
    const [selectedSpec, setSelectedSpec] = useState(null);

    const filteredSpecs = specs.filter(spec => {
        if (confidenceFilter !== 'ALL' && spec.confidence !== confidenceFilter) return false;

        if (filterText) {
            const search = filterText.toLowerCase();
            const inName = spec.item_name?.toLowerCase().includes(search);
            const inText = spec.specification_text?.toLowerCase().includes(search);
            if (!inName && !inText) return false;
        }
        return true;
    });

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
            <div className="controls-bar">
                <div className="input-group">
                    <Search size={18} className="input-icon" />
                    <input
                        type="text"
                        className="text-input"
                        placeholder="Search specs..."
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
                        <option value="ALL">All Confidences</option>
                        <option value="HIGH">High Confidence</option>
                        <option value="MEDIUM">Medium Confidence</option>
                        <option value="LOW">Low Confidence</option>
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
                            <th>Item Name</th>
                            <th>Specification</th>
                            <th>Unit</th>
                            <th>Value</th>
                            <th>Standard</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredSpecs.map((spec, i) => (
                            <tr key={i} onClick={() => setSelectedSpec(spec)}>
                                <td style={{ fontWeight: 500, color: 'var(--text-main)' }}>{spec.item_name}</td>
                                <td style={{ maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {spec.specification_text}
                                </td>
                                <td>{spec.unit !== 'NOT_FOUND' ? spec.unit : '-'}</td>
                                <td>{spec.numeric_value !== 'NOT_FOUND' ? spec.numeric_value : '-'}</td>
                                <td>{spec.standard_reference !== 'NOT_FOUND' ? spec.standard_reference : '-'}</td>
                                <td>
                                    <span className={`badge ${spec.confidence?.toLowerCase() || 'medium'}`}>
                                        {spec.confidence || 'MEDIUM'}
                                    </span>
                                </td>
                            </tr>
                        ))}
                        {filteredSpecs.length === 0 && (
                            <tr>
                                <td colSpan="6" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
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
                title={selectedSpec?.item_name || 'Source Reference'}
            />
        </div>
    );
}

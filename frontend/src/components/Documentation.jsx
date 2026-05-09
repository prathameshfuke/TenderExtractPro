import React, { useState } from 'react';
import { BookOpen, Cpu, Target, FileText, ChevronRight, ArrowLeft, Terminal, ShieldCheck, Zap } from 'lucide-react';

const DOCS_PAGES = [
  {
    id: 'overview',
    title: 'System Overview',
    icon: <BookOpen size={20} />,
    content: (
      <div className="doc-page">
        <h1>System Overview</h1>
        <p className="lead">TenderExtractPro is an industrial-grade intelligence platform designed to automate the extraction and analysis of complex procurement documents.</p>
        
        <div className="doc-section">
          <h3>Architectural Philosophy</h3>
          <p>Unlike standard PDF parsers, our system employs a <strong>Modular Intelligence Pipeline</strong>. We treat every tender as a high-dimensional data structure rather than just text. The system follows a sequence designed for maximum data integrity:</p>
          <ul className="doc-list">
            <li><strong>Ingestion:</strong> Multi-layer parsing including OCR and table structure recovery.</li>
            <li><strong>Contextualization:</strong> Semantic chunking and vector-based indexing.</li>
            <li><strong>Extraction:</strong> Editorial LLM analysis with verbatim grounding.</li>
            <li><strong>Evaluation:</strong> Strategic ranking against company operational parameters.</li>
          </ul>
        </div>

        <div className="doc-grid">
          <div className="doc-card-small">
            <ShieldCheck size={24} color="var(--success)" />
            <h4>Zero-Hallucination</h4>
            <p>Every extracted parameter is mapped back to a verbatim source snippet.</p>
          </div>
          <div className="doc-card-small">
            <Zap size={24} color="#d97706" />
            <h4>High Throughput</h4>
            <p>Parallel processing allows for deep analysis of 100+ page documents in seconds.</p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 'extraction',
    title: 'Extraction Engine',
    icon: <Cpu size={20} />,
    content: (
      <div className="doc-page">
        <h1>The Extraction Engine</h1>
        <p className="lead">At the core of the platform is a hybrid RAG (Retrieval-Augmented Generation) engine optimized for technical precision.</p>

        <div className="doc-section">
          <h3>Stage 1: Multi-Modal Ingestion</h3>
          <p>Documents are ingested using high-performance C-bindings (PyMuPDF). For scanned components, we utilize Tesseract OCR with adaptive thresholding. Table structures are extracted into structured XML before being passed to the LLM to preserve row-column relationships.</p>
        </div>

        <div className="doc-section">
          <h3>Stage 2: Hybrid Retrieval</h3>
          <p>We use a dual-retrieval strategy to ensure technical IDs are never missed:</p>
          <div className="code-block" style={{ background: '#0c0a09', color: '#22d3ee' }}>
            <Terminal size={14} />
            <span>Score = (0.7 * Vector_Sim) + (0.3 * BM25_Keyword)</span>
          </div>
          <p>This ensures that both conceptual matches and specific technical IDs (like "ASTM A36") are captured during the analysis phase.</p>
        </div>

        <div className="doc-section">
          <h3>Stage 3: LLM Reasoning</h3>
          <p>We utilize an 8-bit quantized Mistral-7B model with a 20-year "Procurement Expert" persona. The model performs "Chain-of-Thought" reasoning before outputting final JSON, ensuring it evaluates the context of a specification before committing it as a data point.</p>
        </div>
      </div>
    )
  },
  {
    id: 'pipeline',
    title: 'Pipeline Logic',
    icon: <Terminal size={20} />,
    content: (
      <div className="doc-page">
        <h1>Pipeline Orchestration</h1>
        <p className="lead">Understand the lifecycle of a document from upload to final ranking.</p>

        <div className="doc-section">
          <h3>The Queuing System</h3>
          <p>To ensure system stability, all uploads are handled by a <strong>Sequential Job Worker</strong>. This prevents CPU and VRAM spikes by processing one document at a time while keeping others in a 'Queued' state.</p>
        </div>

        <div className="doc-section">
          <h3>Data ValidationPass</h3>
          <p>Before any data is presented in the UI, it undergoes a JSON validation pass against our strict procurement schemas. If an LLM response is malformed, our <strong>Auto-Repair Logic</strong> attempts to fix brackets or quote nesting without altering the technical values.</p>
        </div>

        <div className="doc-section">
          <h3>Schema Integrity</h3>
          <p>Our extraction target follows the <code>ProcurementStandard-v4</code> schema, focusing on:</p>
          <ul className="doc-list">
            <li><strong>Technical Specs:</strong> Verbatim values and units.</li>
            <li><strong>Scope:</strong> Deliverables, exclusions, and locations.</li>
            <li><strong>Metadata:</strong> Pages, clauses, and confidence intervals.</li>
          </ul>
        </div>
      </div>
    )
  },
  {
    id: 'ranking',
    title: 'Strategic Ranking',
    icon: <Target size={20} />,
    content: (
      <div className="doc-page">
        <h1>Strategic Ranking</h1>
        <p className="lead">The ranking system transforms raw data into strategic insight by evaluating tenders against your unique Company Profile.</p>

        <div className="doc-section">
          <h3>Alignment Logic</h3>
          <p>After extraction, the system runs a secondary inference pass. It compares extracted specifications (Budget, Location, Technical Requirements) against your profile parameters.</p>
          <ul className="doc-list">
            <li><strong>P0 (Direct Match):</strong> >80% alignment. High feasibility and strategic priority.</li>
            <li><strong>P1 (Target):</strong> 60-80% alignment. Strong fit but may require technical deviations.</li>
            <li><strong>P2 (Opportunistic):</strong> &lt;60% alignment. Low feasibility or high risk.</li>
          </ul>
        </div>

        <div className="doc-section">
          <h3>Profile Grounding</h3>
          <p>The ranking isn't just a number—it's backed by <strong>Strategic Reasoning</strong>. The LLM explains <em>why</em> it assigned a specific score, highlighting mismatches in capabilities or budget thresholds.</p>
        </div>
      </div>
    )
  }
];

export default function Documentation({ onBack }) {
  const [activePageId, setActivePageId] = useState('overview');
  
  const activePage = DOCS_PAGES.find(p => p.id === activePageId);

  return (
    <div className="docs-container animate-fade">
      <aside className="docs-sidebar">
        <button className="docs-back" onClick={onBack}>
          <ArrowLeft size={18} />
          Back to Platform
        </button>
        
        <div className="docs-nav">
          <div className="docs-nav-label">Documentation</div>
          {DOCS_PAGES.map(page => (
            <button 
              key={page.id}
              className={`docs-nav-item ${activePageId === page.id ? 'active' : ''}`}
              onClick={() => setActivePageId(page.id)}
            >
              {page.icon}
              {page.title}
              <ChevronRight size={14} className="chevron" />
            </button>
          ))}
        </div>

        <div className="docs-footer">
          <div className="version-tag">Build v2.4.0-PRO</div>
          <p>© 2026 Procurement Intelligence</p>
        </div>
      </aside>

      <main className="docs-content">
        <div className="docs-inner animate-fade" key={activePageId}>
          {activePage.content}
        </div>
      </main>
    </div>
  );
}

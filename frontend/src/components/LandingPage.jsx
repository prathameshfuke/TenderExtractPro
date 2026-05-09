import React from 'react';
import { Shield, Zap, Target, ArrowRight, FileText } from 'lucide-react';
import logo from '../assets/logo.png';

export default function LandingPage({ onGetStarted, onShowDocs }) {
  return (
    <div className="landing-page animate-fade">
      <section className="hero-section">
        <div className="landing-logo animate-scale" style={{ marginBottom: '40px' }}>
          <img src={logo} alt="TenderExtractPro Logo" style={{ height: '80px', width: 'auto' }} />
        </div>
        <div className="hero-tag animate-scale">Next-Gen Procurement Intelligence</div>
        <h1 className="hero-title animate-fade">
          Extract precision from <span style={{ color: 'var(--text-muted)' }}>complexity.</span>
        </h1>
        <p className="hero-subtitle animate-fade" style={{ animationDelay: '0.1s' }}>
          TenderExtractPro uses advanced hybrid retrieval and editorial LLM analysis 
          to transform dense tender PDFs into structured, actionable intelligence.
        </p>
        
        <div className="hero-actions animate-fade" style={{ animationDelay: '0.2s', display: 'flex', gap: '16px' }}>
          <button className="primary-pill" onClick={onGetStarted}>
            Get Started
            <ArrowRight size={18} />
          </button>
          <button className="outline-pill" onClick={onShowDocs}>
            Documentation
          </button>
        </div>

        <div className="feature-grid">
          <div className="feature-card animate-fade" style={{ animationDelay: '0.3s' }}>
            <div className="feature-icon">
              <Target size={24} />
            </div>
            <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: '12px', fontSize: '18px' }}>Technical Extraction</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
              Extract technical specifications and parameters directly from tender documents.
            </p>
          </div>

          <div className="feature-card animate-fade" style={{ animationDelay: '0.4s' }}>
            <div className="feature-icon">
              <Zap size={24} />
            </div>
            <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: '12px', fontSize: '18px' }}>Match Analysis</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
              Evaluate document alignment with company profiles using LLM-based scoring.
            </p>
          </div>

          <div className="feature-card animate-fade" style={{ animationDelay: '0.5s' }}>
            <div className="feature-icon">
              <Shield size={24} />
            </div>
            <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: '12px', fontSize: '18px' }}>Source Tracing</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
              Locate the exact sections and pages used to generate extracted data points.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

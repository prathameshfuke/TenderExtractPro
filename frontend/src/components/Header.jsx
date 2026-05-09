import React from 'react';
import { Plus, User, FileText, LayoutGrid } from 'lucide-react';

export default function Header({ onShowProfile, onUploadClick, onLogoClick, showProfile, hasSelectedJob }) {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-left" onClick={onLogoClick} style={{ cursor: 'pointer' }}>
          <div className="logo-icon">
            <FileText size={20} color="white" />
          </div>
          <div className="brand">
            <span className="brand-name">TenderExtract</span>
            <span className="brand-suffix">Pro</span>
          </div>
        </div>

        <nav className="header-nav">
          <button 
            className={`nav-item ${!hasSelectedJob && !showProfile ? 'active' : ''}`}
            onClick={onLogoClick}
          >
            <LayoutGrid size={18} />
            Workspace
          </button>
        </nav>

        <div className="header-right">
          <button className="primary-pill" onClick={onUploadClick}>
            <Plus size={18} />
            <span>Analyze Tender</span>
          </button>
          
          <div className="v-divider"></div>
          
          <button 
            className={`profile-trigger ${showProfile ? 'active' : ''}`}
            onClick={onShowProfile}
          >
            <User size={20} />
          </button>
        </div>
      </div>
    </header>
  );
}

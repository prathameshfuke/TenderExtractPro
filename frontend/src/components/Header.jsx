import React from 'react';
import { Plus, User, FileText, LayoutGrid, Sun, Moon } from 'lucide-react';
import logo from '../assets/logo.png';

export default function Header({ 
  onShowProfile, 
  onUploadClick, 
  onLogoClick, 
  showProfile, 
  hasSelectedJob,
  theme,
  onToggleTheme,
  onShowDocs,
  showDocs
}) {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-left" onClick={onLogoClick} style={{ cursor: 'pointer' }}>
          <div className="logo-container">
            <img src={logo} alt="Logo" style={{ height: '32px', width: 'auto' }} />
          </div>
          <div className="brand" style={{ letterSpacing: '-0.04em', fontSize: '22px' }}>
            <span className="brand-name" style={{ fontWeight: 800 }}>TenderExtract</span>
            <span className="brand-suffix" style={{ fontWeight: 300, opacity: 0.6 }}>PRO</span>
          </div>
        </div>

        <nav className="header-nav">
          <button 
            className={`nav-item ${!hasSelectedJob && !showProfile && !showDocs ? 'active' : ''}`}
            onClick={onLogoClick}
          >
            <LayoutGrid size={18} />
            Workspace
          </button>
          <button 
            className={`nav-item ${showDocs ? 'active' : ''}`}
            onClick={onShowDocs}
          >
            <BookOpen size={18} />
            Documentation
          </button>
        </nav>


        <div className="header-right">
          <button 
            className="icon-btn" 
            onClick={onToggleTheme} 
            title={theme === 'dark' ? 'Switch to Light' : 'Switch to Dark'}
            style={{ borderRadius: '50%' }}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          <div className="v-divider"></div>

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

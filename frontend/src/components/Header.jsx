import React from 'react';
import { Plus, User, FileText, LayoutGrid, Sun, Moon } from 'lucide-react';

export default function Header({ 
  onShowProfile, 
  onUploadClick, 
  onLogoClick, 
  showProfile, 
  hasSelectedJob,
  theme,
  onToggleTheme 
}) {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-left" onClick={onLogoClick} style={{ cursor: 'pointer' }}>
          <div className="logo-icon">
            <FileText size={20} color={theme === 'dark' ? '#0c0a09' : 'white'} />
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

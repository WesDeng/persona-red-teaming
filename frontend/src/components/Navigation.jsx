/**
 * Navigation component
 *
 * Provides tab navigation between Home, Persona History, and Prompt History views.
 * Displays session statistics in the navigation bar.
 */

import React from 'react';

const Navigation = ({ currentView, setCurrentView, stats }) => {
  const tabs = [
    { id: 'home', label: 'Home' },
    { id: 'persona-history', label: 'Persona History' },
    { id: 'prompt-history', label: 'Prompt History' },
  ];

  return (
    <nav className="navigation">
      <div className="nav-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${currentView === tab.id ? 'active' : ''}`}
            onClick={() => setCurrentView(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {stats && (
        <div className="nav-stats">
          <div className="stat-item">
            <span className="stat-label">Personas:</span>
            <span className="stat-value">{stats.total_personas}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Prompts:</span>
            <span className="stat-value">{stats.total_prompts}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Success Rate:</span>
            <span className="stat-value">{stats.overall_success_rate.toFixed(1)}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Unsafe (Guard):</span>
            <span className="stat-value">{stats.total_unsafe_by_guard}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Unsafe (User):</span>
            <span className="stat-value">{stats.total_unsafe_by_user}</span>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navigation;

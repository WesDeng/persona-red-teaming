/**
 * HistorySidebar component
 *
 * Collapsible sidebar showing recent personas and prompts.
 * Provides quick access to recent history without leaving the main view.
 */

import React, { useState, useEffect } from 'react';
import { getPersonaHistoryAPI, getPromptHistoryAPI } from '../utils/api';

const HistorySidebar = ({ isOpen, onClose, onLoadPersona, onViewPrompt }) => {
  const [sidebarTab, setSidebarTab] = useState('personas'); // 'personas' or 'prompts'
  const [recentPersonas, setRecentPersonas] = useState([]);
  const [recentPrompts, setRecentPrompts] = useState([]);
  const [loading, setLoading] = useState(false);

  // Load recent data when sidebar opens
  useEffect(() => {
    if (isOpen) {
      loadRecentData();
    }
  }, [isOpen]);

  const loadRecentData = async () => {
    setLoading(true);
    try {
      // Load recent 5 personas and prompts
      const [personasRes, promptsRes] = await Promise.all([
        getPersonaHistoryAPI({ limit: 5, sort: 'recent' }),
        getPromptHistoryAPI({ limit: 5 }),
      ]);

      setRecentPersonas(personasRes.data.personas);
      setRecentPrompts(promptsRes.data.prompts);
    } catch (error) {
      console.error('Failed to load recent history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadPersona = (persona) => {
    if (onLoadPersona) {
      onLoadPersona(persona);
    }
    onClose();
  };

  const handleViewPrompt = (prompt) => {
    if (onViewPrompt) {
      onViewPrompt(prompt);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const truncateText = (text, maxLength = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  if (!isOpen) return null;

  return (
    <div className="history-sidebar-overlay" onClick={onClose}>
      <div className="history-sidebar" onClick={(e) => e.stopPropagation()}>
        <div className="sidebar-header">
          <h2>Recent History</h2>
          <button className="close-btn" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="sidebar-tabs">
          <button
            className={`sidebar-tab ${sidebarTab === 'personas' ? 'active' : ''}`}
            onClick={() => setSidebarTab('personas')}
          >
            Personas ({recentPersonas.length})
          </button>
          <button
            className={`sidebar-tab ${sidebarTab === 'prompts' ? 'active' : ''}`}
            onClick={() => setSidebarTab('prompts')}
          >
            Prompts ({recentPrompts.length})
          </button>
        </div>

        <div className="sidebar-content">
          {loading ? (
            <div className="loading">Loading...</div>
          ) : sidebarTab === 'personas' ? (
            <div className="personas-list">
              {recentPersonas.length === 0 ? (
                <div className="empty-state">No personas yet</div>
              ) : (
                recentPersonas.map((persona) => (
                  <div key={persona.id} className="persona-item">
                    <div className="persona-preview">
                      <pre>{truncateText(persona.persona_yaml, 80)}</pre>
                    </div>
                    <div className="persona-meta">
                      <span className="success-rate">
                        {persona.success_rate.toFixed(1)}% success
                      </span>
                      <span className="timestamp">{formatDate(persona.created_at)}</span>
                    </div>
                    <button
                      className="load-btn"
                      onClick={() => handleLoadPersona(persona)}
                    >
                      Load
                    </button>
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="prompts-list">
              {recentPrompts.length === 0 ? (
                <div className="empty-state">No prompts yet</div>
              ) : (
                recentPrompts.map((prompt) => (
                  <div
                    key={prompt.id}
                    className="prompt-item"
                    onClick={() => handleViewPrompt(prompt)}
                  >
                    <div className="prompt-text">
                      {truncateText(prompt.prompt_text, 100)}
                    </div>
                    <div className="prompt-meta">
                      <span
                        className={`verdict ${
                          prompt.guard_verdict === 'unsafe' ? 'unsafe' : 'safe'
                        }`}
                      >
                        {prompt.guard_verdict || 'N/A'}
                      </span>
                      <span className="timestamp">{formatDate(prompt.created_at)}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <button className="view-all-btn" onClick={() => {
            // Switch to full history view
            if (sidebarTab === 'personas') {
              window.dispatchEvent(new CustomEvent('navigate', { detail: 'persona-history' }));
            } else {
              window.dispatchEvent(new CustomEvent('navigate', { detail: 'prompt-history' }));
            }
            onClose();
          }}>
            View All {sidebarTab === 'personas' ? 'Personas' : 'Prompts'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default HistorySidebar;

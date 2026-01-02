/**
 * PersonaHistory component
 *
 * Full page view of persona history with search, sort, and pagination.
 * Displays success metrics and allows loading personas back into the editor.
 */

import React, { useState, useEffect } from 'react';
import { getPersonaHistoryAPI, getPersonaByIdAPI } from '../utils/api';

const PersonaHistory = ({ onLoadPersona }) => {
  const [personas, setPersonas] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  // Pagination and filters
  const [page, setPage] = useState(0);
  const [limit] = useState(20);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('recent'); // 'recent', 'success_rate', 'most_used'

  // Load personas on mount and when filters change
  useEffect(() => {
    loadPersonas();
  }, [page, sortBy]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (page === 0) {
        loadPersonas();
      } else {
        setPage(0); // Reset to first page on search
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  const loadPersonas = async () => {
    setLoading(true);
    try {
      const response = await getPersonaHistoryAPI({
        offset: page * limit,
        limit: limit,
        search: searchQuery || undefined,
        sort: sortBy,
      });

      setPersonas(response.data.personas);
      setTotal(response.data.total);
    } catch (error) {
      console.error('Failed to load persona history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadPersona = async (personaId) => {
    try {
      const response = await getPersonaByIdAPI(personaId);
      const persona = response.data;

      if (onLoadPersona) {
        onLoadPersona({
          personaText: persona.persona_yaml,
          emphasisInstructions: persona.emphasis_instructions || '',
          mutationType: persona.mutation_type || 'persona',
          riskCategory: persona.risk_category || '',
          attackStyle: persona.attack_style || '',
        });
      }
    } catch (error) {
      console.error('Failed to load persona:', error);
      alert('Failed to load persona. Please try again.');
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="persona-history-page">
      <div className="history-header">
        <h1>Persona History</h1>
        <div className="history-controls">
          <input
            type="text"
            className="search-input"
            placeholder="Search personas..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          <select
            className="sort-select"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          >
            <option value="recent">Most Recent</option>
            <option value="success_rate">Highest Success Rate</option>
            <option value="most_used">Most Used</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading personas...</div>
      ) : personas.length === 0 ? (
        <div className="empty-state">
          {searchQuery ? 'No personas match your search.' : 'No personas yet. Create one to get started!'}
        </div>
      ) : (
        <>
          <div className="personas-grid">
            {personas.map((persona) => (
              <div key={persona.id} className="persona-card">
                <div className="persona-header">
                  <div className="persona-type-badge">
                    {persona.mutation_type || 'persona'}
                  </div>
                  <div className="persona-date">{formatDate(persona.created_at)}</div>
                </div>

                <div className="persona-content">
                  <pre className="persona-yaml">{persona.persona_yaml}</pre>

                  {persona.emphasis_instructions && (
                    <div className="emphasis-section">
                      <strong>Emphasis:</strong>
                      <div>{persona.emphasis_instructions}</div>
                    </div>
                  )}

                  {(persona.risk_category || persona.attack_style) && (
                    <div className="mutation-details">
                      {persona.risk_category && (
                        <span className="detail-badge">Risk: {persona.risk_category}</span>
                      )}
                      {persona.attack_style && (
                        <span className="detail-badge">Style: {persona.attack_style}</span>
                      )}
                    </div>
                  )}
                </div>

                <div className="persona-metrics">
                  <div className="metric">
                    <div className="metric-label">Success Rate</div>
                    <div className={`metric-value ${persona.success_rate > 50 ? 'high' : 'low'}`}>
                      {persona.success_rate.toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Generations</div>
                    <div className="metric-value">{persona.total_generations}</div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Unsafe / Safe</div>
                    <div className="metric-value">
                      {persona.unsafe_count} / {persona.safe_count}
                    </div>
                  </div>
                </div>

                <div className="persona-actions">
                  <button
                    className="load-persona-btn"
                    onClick={() => handleLoadPersona(persona.id)}
                  >
                    Load Persona
                  </button>
                </div>
              </div>
            ))}
          </div>

          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="page-btn"
                onClick={() => setPage(Math.max(0, page - 1))}
                disabled={page === 0}
              >
                Previous
              </button>

              <span className="page-info">
                Page {page + 1} of {totalPages} ({total} total)
              </span>

              <button
                className="page-btn"
                onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                disabled={page >= totalPages - 1}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default PersonaHistory;

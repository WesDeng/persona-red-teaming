/**
 * PromptHistory component
 *
 * Full page view of prompt history with filtering and detail panel.
 * Shows guard evaluations, edit history, and allows marking as unsafe.
 */

import React, { useState, useEffect } from 'react';
import { getPromptHistoryAPI, getPromptEditHistoryAPI, markUnsafeAPI } from '../utils/api';

const PromptHistory = () => {
  const [prompts, setPrompts] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  // Pagination and filters
  const [page, setPage] = useState(0);
  const [limit] = useState(50);
  const [verdictFilter, setVerdictFilter] = useState('all'); // 'all', 'safe', 'unsafe'

  // Detail panel
  const [selectedPrompt, setSelectedPrompt] = useState(null);
  const [editHistory, setEditHistory] = useState(null);
  const [loadingEdits, setLoadingEdits] = useState(false);

  // Load prompts on mount and when filters change
  useEffect(() => {
    loadPrompts();
  }, [page, verdictFilter]);

  const loadPrompts = async () => {
    setLoading(true);
    try {
      const response = await getPromptHistoryAPI({
        offset: page * limit,
        limit: limit,
        verdict_filter: verdictFilter,
      });

      setPrompts(response.data.prompts);
      setTotal(response.data.total);
    } catch (error) {
      console.error('Failed to load prompt history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePromptClick = async (prompt) => {
    setSelectedPrompt(prompt);

    // Load edit history if this prompt has edits
    if (prompt.edit_count > 0) {
      setLoadingEdits(true);
      try {
        const response = await getPromptEditHistoryAPI(prompt.id);
        setEditHistory(response.data);
      } catch (error) {
        console.error('Failed to load edit history:', error);
        setEditHistory(null);
      } finally {
        setLoadingEdits(false);
      }
    } else {
      setEditHistory(null);
    }
  };

  const handleMarkUnsafe = async (promptId, currentlyMarked) => {
    try {
      await markUnsafeAPI(promptId, !currentlyMarked);

      // Update local state
      setPrompts((prev) =>
        prev.map((p) =>
          p.id === promptId
            ? { ...p, user_marked_unsafe: !currentlyMarked }
            : p
        )
      );

      // Update selected prompt if it's the one being marked
      if (selectedPrompt && selectedPrompt.id === promptId) {
        setSelectedPrompt((prev) => ({
          ...prev,
          user_marked_unsafe: !currentlyMarked,
        }));
      }
    } catch (error) {
      console.error('Failed to mark unsafe:', error);
      alert('Failed to update feedback. Please try again.');
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="prompt-history-page">
      <div className="history-header">
        <h1>Prompt History</h1>
        <div className="history-controls">
          <div className="filter-buttons">
            <button
              className={`filter-btn ${verdictFilter === 'all' ? 'active' : ''}`}
              onClick={() => {
                setVerdictFilter('all');
                setPage(0);
              }}
            >
              All
            </button>
            <button
              className={`filter-btn ${verdictFilter === 'safe' ? 'active' : ''}`}
              onClick={() => {
                setVerdictFilter('safe');
                setPage(0);
              }}
            >
              Safe
            </button>
            <button
              className={`filter-btn ${verdictFilter === 'unsafe' ? 'active' : ''}`}
              onClick={() => {
                setVerdictFilter('unsafe');
                setPage(0);
              }}
            >
              Unsafe
            </button>
          </div>
        </div>
      </div>

      <div className="prompt-history-content">
        {/* Prompts List */}
        <div className="prompts-list-section">
          {loading ? (
            <div className="loading">Loading prompts...</div>
          ) : prompts.length === 0 ? (
            <div className="empty-state">
              {verdictFilter !== 'all'
                ? `No ${verdictFilter} prompts found.`
                : 'No prompts yet. Generate some to get started!'}
            </div>
          ) : (
            <>
              <div className="prompts-table">
                <div className="table-header">
                  <div className="col-type">Type</div>
                  <div className="col-prompt">Prompt</div>
                  <div className="col-verdict">Verdict</div>
                  <div className="col-score">Score</div>
                  <div className="col-edits">Edits</div>
                  <div className="col-date">Date</div>
                  <div className="col-actions">Actions</div>
                </div>

                {prompts.map((prompt) => (
                  <div
                    key={prompt.id}
                    className={`table-row ${selectedPrompt?.id === prompt.id ? 'selected' : ''}`}
                    onClick={() => handlePromptClick(prompt)}
                  >
                    <div className="col-type">
                      <span className={`type-badge ${prompt.prompt_type}`}>
                        {prompt.prompt_type}
                      </span>
                    </div>
                    <div className="col-prompt">
                      <div className="prompt-preview">
                        {prompt.prompt_text.substring(0, 100)}
                        {prompt.prompt_text.length > 100 && '...'}
                      </div>
                    </div>
                    <div className="col-verdict">
                      {prompt.guard_verdict ? (
                        <span className={`verdict-badge ${prompt.guard_verdict}`}>
                          {prompt.guard_verdict}
                        </span>
                      ) : (
                        <span className="verdict-badge unknown">N/A</span>
                      )}
                    </div>
                    <div className="col-score">
                      {prompt.guard_score !== null
                        ? prompt.guard_score.toFixed(2)
                        : 'N/A'}
                    </div>
                    <div className="col-edits">{prompt.edit_count}</div>
                    <div className="col-date">{formatDate(prompt.created_at)}</div>
                    <div className="col-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        className={`mark-unsafe-btn ${prompt.user_marked_unsafe ? 'marked' : ''}`}
                        onClick={() =>
                          handleMarkUnsafe(prompt.id, prompt.user_marked_unsafe)
                        }
                        title={
                          prompt.user_marked_unsafe
                            ? 'Unmark as unsafe'
                            : 'Mark as unsafe'
                        }
                      >
                        {prompt.user_marked_unsafe ? '⚠' : '○'}
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

        {/* Detail Panel */}
        {selectedPrompt && (
          <div className="prompt-detail-panel">
            <div className="panel-header">
              <h2>Prompt Details</h2>
              <button className="close-panel-btn" onClick={() => setSelectedPrompt(null)}>
                ✕
              </button>
            </div>

            <div className="panel-content">
              {/* Prompt Text */}
              <section className="detail-section">
                <h3>Prompt Text</h3>
                <div className="prompt-full-text">{selectedPrompt.prompt_text}</div>
              </section>

              {/* Seed Prompt */}
              {selectedPrompt.seed_prompt_text && (
                <section className="detail-section">
                  <h3>Seed Prompt</h3>
                  <div className="seed-prompt-text">
                    {selectedPrompt.seed_prompt_text}
                  </div>
                </section>
              )}

              {/* Target Response */}
              {selectedPrompt.target_response && (
                <section className="detail-section">
                  <h3>Target Response</h3>
                  <div className="target-response-text">
                    {selectedPrompt.target_response}
                  </div>
                </section>
              )}

              {/* Guard Evaluation */}
              <section className="detail-section">
                <h3>Guard Evaluation</h3>
                <div className="guard-evaluation">
                  <div className="eval-item">
                    <span className="eval-label">Verdict:</span>
                    <span className={`eval-value verdict ${selectedPrompt.guard_verdict}`}>
                      {selectedPrompt.guard_verdict || 'N/A'}
                    </span>
                  </div>
                  <div className="eval-item">
                    <span className="eval-label">Score:</span>
                    <span className="eval-value">
                      {selectedPrompt.guard_score !== null
                        ? selectedPrompt.guard_score.toFixed(2)
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="eval-item">
                    <span className="eval-label">User Marked Unsafe:</span>
                    <span className="eval-value">
                      {selectedPrompt.user_marked_unsafe ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
              </section>

              {/* Edit History */}
              {selectedPrompt.edit_count > 0 && (
                <section className="detail-section">
                  <h3>Edit History ({selectedPrompt.edit_count} edits)</h3>
                  {loadingEdits ? (
                    <div className="loading-edits">Loading edit history...</div>
                  ) : editHistory ? (
                    <div className="edit-timeline">
                      {editHistory.edits.map((edit, index) => (
                        <div key={edit.id} className="timeline-item">
                          <div className="timeline-marker">
                            {edit.is_original ? '●' : '○'}
                          </div>
                          <div className="timeline-content">
                            <div className="timeline-header">
                              <span className="timeline-label">
                                {edit.is_original ? 'Original' : `Edit ${index}`}
                              </span>
                              <span className="timeline-date">
                                {formatDate(edit.created_at)}
                              </span>
                            </div>
                            <div className="timeline-text">{edit.prompt_text}</div>
                            <div className="timeline-eval">
                              <span className={`verdict ${edit.guard_verdict}`}>
                                {edit.guard_verdict}
                              </span>
                              <span className="score">
                                Score: {edit.guard_score.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="edit-error">Failed to load edit history</div>
                  )}
                </section>
              )}

              {/* Persona Used */}
              <section className="detail-section">
                <h3>Persona Used</h3>
                <pre className="persona-yaml-text">{selectedPrompt.persona_yaml}</pre>
              </section>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PromptHistory;

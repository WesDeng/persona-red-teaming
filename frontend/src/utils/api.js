/**
 * API client with axios interceptor for session management
 *
 * This module provides:
 * - Configured axios instance with session ID header injection
 * - Typed API functions for all backend endpoints
 * - Centralized error handling
 *
 * Usage:
 *   import { generateAPI, getPersonaHistoryAPI } from './utils/api';
 *   const result = await generateAPI(requestData);
 */

import axios from 'axios';
import { getSessionId } from './session';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Create axios instance with base configuration
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for generation endpoints
});

/**
 * Request interceptor: Add session ID to all requests
 */
apiClient.interceptors.request.use(
  (config) => {
    const sessionId = getSessionId();
    config.headers['X-Session-ID'] = sessionId;
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor: Handle common errors
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('Network Error: No response received');
    } else {
      console.error('Request Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// ============================================================================
// GENERATION ENDPOINTS
// ============================================================================

/**
 * Generate adversarial prompts from persona
 * POST /api/generate
 *
 * @param {Object} requestData - Generation request data
 * @param {string} requestData.persona_text - YAML persona definition
 * @param {string} requestData.seed_mode - 'random' or 'preselected'
 * @param {number} requestData.num_seed_prompts - Number of seed prompts
 * @param {number} requestData.num_mutations_per_seed - Mutations per seed
 * @param {string} [requestData.emphasis_instructions] - Optional emphasis
 * @param {string} [requestData.mutation_type] - 'persona', 'rainbow', 'risk-category'
 * @param {string} [requestData.risk_category] - For rainbow/risk-category types
 * @param {string} [requestData.attack_style] - For rainbow/risk-category types
 * @returns {Promise<Object>} Generated results with prompts and evaluations
 */
export const generateAPI = (requestData) => {
  return apiClient.post('/api/generate', requestData);
};

/**
 * Reattack with edited prompt
 * POST /api/reattack
 *
 * @param {Object} requestData - Reattack request data
 * @param {string} requestData.prompt - Edited prompt text
 * @param {string} requestData.persona_text - YAML persona definition
 * @param {number} [originalPromptId] - Original prompt ID for edit history
 * @returns {Promise<Object>} Reattack result with evaluation
 */
export const reattackAPI = (requestData, originalPromptId = null) => {
  const params = originalPromptId ? { original_prompt_id: originalPromptId } : {};
  return apiClient.post('/api/reattack', requestData, { params });
};

/**
 * Get preselected seed prompts
 * GET /api/preselected-seeds
 *
 * @returns {Promise<Array<string>>} List of preselected seed prompts
 */
export const getPreselectedSeedsAPI = () => {
  return apiClient.get('/api/preselected-seeds');
};

/**
 * Refresh preselected seed prompts
 * POST /api/preselected-seeds/refresh
 *
 * @returns {Promise<Object>} Refresh confirmation with new count
 */
export const refreshPreselectedSeedsAPI = () => {
  return apiClient.post('/api/preselected-seeds/refresh');
};

/**
 * Get persona mutation suggestions
 * POST /api/suggest-mutations
 *
 * @param {Object} requestData - Suggestion request
 * @param {string} requestData.persona_text - Current YAML persona definition
 * @returns {Promise<Object>} Mutation suggestions
 */
export const getSuggestMutationsAPI = (requestData) => {
  return apiClient.post('/api/suggest-mutations', requestData);
};

// ============================================================================
// USER FEEDBACK ENDPOINTS
// ============================================================================

/**
 * Mark prompt as unsafe (user feedback)
 * POST /api/mark-unsafe
 *
 * @param {number} promptId - Prompt ID to mark
 * @param {boolean} marked - True to mark as unsafe, false to unmark
 * @returns {Promise<Object>} Feedback save confirmation
 */
export const markUnsafeAPI = (promptId, marked) => {
  return apiClient.post('/api/mark-unsafe', {
    prompt_id: promptId,
    marked: marked,
  });
};

// ============================================================================
// HISTORY ENDPOINTS
// ============================================================================

/**
 * Get persona history with success metrics
 * GET /api/history/personas
 *
 * @param {Object} [params] - Query parameters
 * @param {number} [params.offset=0] - Pagination offset
 * @param {number} [params.limit=20] - Items per page
 * @param {string} [params.search] - Search by persona text
 * @param {string} [params.sort='recent'] - Sort by: 'recent', 'success_rate', 'most_used'
 * @returns {Promise<Object>} PersonaHistoryResponse with personas and total count
 */
export const getPersonaHistoryAPI = (params = {}) => {
  return apiClient.get('/api/history/personas', { params });
};

/**
 * Get single persona by ID
 * GET /api/history/personas/{id}
 *
 * @param {number} personaId - Persona ID
 * @returns {Promise<Object>} Persona details
 */
export const getPersonaByIdAPI = (personaId) => {
  return apiClient.get(`/api/history/personas/${personaId}`);
};

/**
 * Get prompt history with guard results
 * GET /api/history/prompts
 *
 * @param {Object} [params] - Query parameters
 * @param {number} [params.offset=0] - Pagination offset
 * @param {number} [params.limit=50] - Items per page
 * @param {string} [params.verdict_filter] - Filter by: 'all', 'safe', 'unsafe'
 * @returns {Promise<Object>} PromptHistoryResponse with prompts and total count
 */
export const getPromptHistoryAPI = (params = {}) => {
  return apiClient.get('/api/history/prompts', { params });
};

/**
 * Get edit history for a prompt
 * GET /api/history/prompts/{id}/edits
 *
 * @param {number} promptId - Prompt ID
 * @returns {Promise<Object>} EditHistoryResponse with edit chain
 */
export const getPromptEditHistoryAPI = (promptId) => {
  return apiClient.get(`/api/history/prompts/${promptId}/edits`);
};

/**
 * Get session statistics
 * GET /api/history/stats
 *
 * @returns {Promise<Object>} SessionStats with aggregate metrics
 */
export const getSessionStatsAPI = () => {
  return apiClient.get('/api/history/stats');
};

/**
 * Delete current session and all data
 * DELETE /api/session
 *
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteSessionAPI = () => {
  return apiClient.delete('/api/session');
};

// ============================================================================
// EXPORT DEFAULT CLIENT
// ============================================================================

export default apiClient;

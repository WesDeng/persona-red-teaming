/**
 * Session management utilities
 *
 * Handles anonymous session tracking via localStorage.
 * Each browser gets a unique UUID that persists across page refreshes.
 *
 * Usage:
 *   import { getSessionId } from './utils/session';
 *   const sessionId = getSessionId(); // Gets or creates session ID
 */

const SESSION_KEY = 'persona-rt-session-id';

/**
 * Generate a UUID v4
 *
 * @returns {string} UUID in format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
 */
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Get or create session ID
 *
 * Checks localStorage for existing session ID. If none exists,
 * generates a new UUID and stores it.
 *
 * @returns {string} Session ID (UUID)
 */
export function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);

  if (!sessionId) {
    sessionId = generateUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
    console.log('Created new session:', sessionId);
  }

  return sessionId;
}

/**
 * Clear session (for testing or user-initiated deletion)
 *
 * Removes session ID from localStorage. Next call to getSessionId()
 * will create a new session.
 */
export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
  console.log('Session cleared');
}

/**
 * Check if a session exists
 *
 * @returns {boolean} True if session ID exists in localStorage
 */
export function hasSession() {
  return localStorage.getItem(SESSION_KEY) !== null;
}

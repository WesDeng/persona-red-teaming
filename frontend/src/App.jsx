import React, { useState, useEffect } from 'react'
import Navigation from './components/Navigation'
import HistorySidebar from './components/HistorySidebar'
import PersonaHistory from './components/PersonaHistory'
import PromptHistory from './components/PromptHistory'
import {
  generateAPI,
  reattackAPI,
  getPreselectedSeedsAPI,
  refreshPreselectedSeedsAPI,
  getSuggestMutationsAPI,
  markUnsafeAPI,
  getSessionStatsAPI,
} from './utils/api'

const INITIAL_PERSONA = `name: [name]
age: [age]
sex: [sex]
ethnicity: [ethnicity]
race: [race]
hispanic_origin: [hispanic_origin]
city: [city]
state: [state]
political_views: [political_views]
religion: [religion]
party_identification: [party_identification]
total_wealth: [total_wealth]
background: [background]
location: [location]
behavioral_traits:
  - [behavioral_trait_1]
  - [behavioral_trait_2]
  - [behavioral_trait_3]
tech_use: [tech_use]`

function App() {
  // Navigation and history state
  const [currentView, setCurrentView] = useState('home') // 'home', 'persona-history', 'prompt-history'
  const [stats, setStats] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Persona editor state
  const [persona, setPersona] = useState('')
  const [emphasisInstructions, setEmphasisInstructions] = useState('')
  const [numSeedPrompts, setNumSeedPrompts] = useState(5)
  const [numMutations, setNumMutations] = useState(3)
  const [seedMode, setSeedMode] = useState('random') // 'random' or 'preselected'
  const [preselectedSeeds, setPreselectedSeeds] = useState([])
  const [selectedSeeds, setSelectedSeeds] = useState([]) // User-selected seeds
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [editingPrompt, setEditingPrompt] = useState(null) // {resultIdx, promptIdx}
  const [editedPromptText, setEditedPromptText] = useState('')
  const [reattacking, setReattacking] = useState(null) // {resultIdx, promptIdx}
  const [showInstructions, setShowInstructions] = useState(false)
  const [emphasisExpanded, setEmphasisExpanded] = useState(false)
  const [mutationParametersExpanded, setMutationParametersExpanded] = useState(false)
  const [riskCategory, setRiskCategory] = useState('')
  const [attackStyle, setAttackStyle] = useState('')
  const [mutationType, setMutationType] = useState('persona') // 'persona', 'rainbow', 'risk-category'
  const [markedUnsafe, setMarkedUnsafe] = useState(new Set()) // Track user-marked unsafe items
  const [refreshingSeeds, setRefreshingSeeds] = useState(false)
  const [showingSuggestionsFor, setShowingSuggestionsFor] = useState(null) // {resultIdx, promptIdx}
  const [currentSuggestions, setCurrentSuggestions] = useState([])
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [baselineMode, setBaselineMode] = useState(false) // Baseline mode without persona

  const personaInstructions = [
    'We are not storing any of your data. All the data is processed locally in your browser.',
    'Highlight your unique characteristics to make your persona unique!',
    'Provide concrete behavioral traits using lists for clarity.',
    'Keep descriptions concise; aim for 3-5 sentences per section.',
    'You can iteratively edit and re-attack the prompts to find the most effective adversarial prompts.',
    'You can also generate random prompts to get a sense of the diversity of the prompts.',
    "You can also directly edit the adversarial prompts to make them more effective!"
  ]

  // Load session stats on mount
  useEffect(() => {
    loadStats()

    // Listen for custom navigation events from sidebar
    const handleNavigate = (event) => {
      setCurrentView(event.detail)
    }
    window.addEventListener('navigate', handleNavigate)
    return () => window.removeEventListener('navigate', handleNavigate)
  }, [])

  const loadStats = async () => {
    try {
      const response = await getSessionStatsAPI()
      setStats(response.data)
    } catch (error) {
      console.error('Failed to load session stats:', error)
      // Set default stats if none exist
      setStats({
        total_personas: 0,
        total_prompts: 0,
        total_unsafe_by_guard: 0,
        total_unsafe_by_user: 0,
        overall_success_rate: 0,
      })
    }
  }

  // Fetch preselected seeds when mode changes
  useEffect(() => {
    if (seedMode === 'preselected' && preselectedSeeds.length === 0) {
      getPreselectedSeedsAPI()
        .then(response => {
          setPreselectedSeeds(response.data.seeds)
        })
        .catch(err => {
          console.error('Failed to fetch preselected seeds:', err)
        })
    }
  }, [seedMode, preselectedSeeds.length])

  const handleSeedToggle = (seed) => {
    setSelectedSeeds(prev => {
      if (prev.includes(seed)) {
        return prev.filter(s => s !== seed)
      } else {
        return [...prev, seed]
      }
    })
  }

  const handleRefreshSeeds = async () => {
    setRefreshingSeeds(true)
    try {
      const response = await refreshPreselectedSeedsAPI()
      setPreselectedSeeds(response.data.seeds)
      // Clear current selections since we have new seeds
      setSelectedSeeds([])
    } catch (err) {
      console.error('Failed to refresh seeds:', err)
      setError('Failed to refresh seed prompts')
    } finally {
      setRefreshingSeeds(false)
    }
  }

  const handleGetSuggestions = async (resultIdx, promptIdx) => {
    setLoadingSuggestions(true)
    setShowingSuggestionsFor({ resultIdx, promptIdx })
    setError(null)

    try {
      const result = results[resultIdx]
      const response = await getSuggestMutationsAPI({
        seed_prompt: result.seed_prompt,
        adversarial_prompt: result.adversarial_prompts[promptIdx],
        persona: persona
      })

      setCurrentSuggestions(response.data.suggestions)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate suggestions')
      setShowingSuggestionsFor(null)
    } finally {
      setLoadingSuggestions(false)
    }
  }

  const handleCloseSuggestions = () => {
    setShowingSuggestionsFor(null)
    setCurrentSuggestions([])
  }

  const handleApplyTemplate = () => {
    setPersona(INITIAL_PERSONA)
  }

  const handleMarkUnsafe = async (resultIdx, promptIdx, promptId) => {
    const key = `${resultIdx}-${promptIdx}`
    const newMarked = !markedUnsafe.has(key)

    // Update local state optimistically
    setMarkedUnsafe(prev => {
      const newSet = new Set(prev)
      if (newSet.has(key)) {
        newSet.delete(key)
      } else {
        newSet.add(key)
      }
      return newSet
    })

    // Save to backend if we have a prompt ID
    if (promptId) {
      try {
        await markUnsafeAPI(promptId, newMarked)
        // Reload stats after marking
        loadStats()
      } catch (error) {
        console.error('Failed to save feedback:', error)
        // Revert on error
        setMarkedUnsafe(prev => {
          const newSet = new Set(prev)
          if (newSet.has(key)) {
            newSet.delete(key)
          } else {
            newSet.add(key)
          }
          return newSet
        })
      }
    }
  }

  const handleGenerate = async () => {
    // Validation for preselected mode
    if (seedMode === 'preselected' && selectedSeeds.length === 0) {
      setError('Please select at least one seed prompt')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)
    setBaselineMode(false) // Exit baseline mode when generating with persona

    try {
      const requestData = {
        persona: persona,
        emphasis_instructions: emphasisInstructions,
        risk_category: riskCategory,
        attack_style: attackStyle,
        mutation_type: mutationType,
        num_seed_prompts: numSeedPrompts,
        num_mutations_per_seed: numMutations,
        seed_mode: seedMode
      }

      // Add selected seeds for preselected mode
      if (seedMode === 'preselected') {
        requestData.selected_seeds = selectedSeeds
      }

      const response = await generateAPI(requestData)

      setResults(response.data.results)
      // Reload stats after successful generation
      loadStats()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate prompts')
    } finally {
      setLoading(false)
    }
  }

  const handleBaseline = async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    setBaselineMode(true) // Enter baseline mode

    try {
      const requestData = {
        persona: '', // No persona for baseline
        emphasis_instructions: '',
        risk_category: '',
        attack_style: '',
        mutation_type: 'persona', // Use default mutation type
        num_seed_prompts: 3, // Fixed 3 seed prompts for baseline
        num_mutations_per_seed: 0, // No automatic mutations, user will mutate manually
        seed_mode: 'random'
      }

      const response = await generateAPI(requestData)

      setResults(response.data.results)
      // Reload stats after successful generation
      loadStats()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate baseline prompts')
    } finally {
      setLoading(false)
    }
  }

  const handleEditPrompt = (resultIdx, promptIdx) => {
    const prompt = results[resultIdx].adversarial_prompts[promptIdx]
    setEditedPromptText(prompt)
    setEditingPrompt({ resultIdx, promptIdx })
  }

  const handleCancelEdit = () => {
    setEditingPrompt(null)
    setEditedPromptText('')
  }

  const handleSaveAndReattack = async () => {
    const { resultIdx, promptIdx } = editingPrompt
    setReattacking({ resultIdx, promptIdx })
    setError(null)

    try {
      // Get original prompt ID if available
      const originalPromptId = results[resultIdx].prompt_ids?.[promptIdx]

      const response = await reattackAPI(
        { prompt: editedPromptText },
        originalPromptId
      )

      // Update the results with new response and guard result
      const newResults = [...results]
      newResults[resultIdx].adversarial_prompts[promptIdx] = editedPromptText
      newResults[resultIdx].target_responses[promptIdx] = response.data.target_response
      newResults[resultIdx].guard_results[promptIdx] = response.data.guard_result

      // Update prompt ID if a new one was returned
      if (response.data.prompt_id && newResults[resultIdx].prompt_ids) {
        newResults[resultIdx].prompt_ids[promptIdx] = response.data.prompt_id
      }

      setResults(newResults)

      setEditingPrompt(null)
      setEditedPromptText('')

      // Reload stats after reattack
      loadStats()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to reattack')
    } finally {
      setReattacking(null)
    }
  }

  // Handler for loading persona from history
  const handleLoadPersona = (personaData) => {
    setPersona(personaData.personaText || personaData.persona_yaml || '')
    setEmphasisInstructions(personaData.emphasisInstructions || personaData.emphasis_instructions || '')
    setMutationType(personaData.mutationType || personaData.mutation_type || 'persona')
    setRiskCategory(personaData.riskCategory || personaData.risk_category || '')
    setAttackStyle(personaData.attackStyle || personaData.attack_style || '')
    setCurrentView('home')
    setSidebarOpen(false)
  }

  return (
    <div className="app">
      <Navigation
        currentView={currentView}
        setCurrentView={setCurrentView}
        stats={stats}
      />

      {currentView === 'home' && (
        <>
          <div className="header">
            <div className="header-content">
              <h1>PersonaTeaming Interactive Interface</h1>
              <button
                className="history-btn"
                onClick={() => setSidebarOpen(!sidebarOpen)}
              >
                {sidebarOpen ? 'Close History' : 'View History'}
              </button>
            </div>
            <p>This is an interactive interface where you can draft and improve your own persona to generate adversarial prompts. <br /> You can also iteratively edit and re-attack the prompts to find the most effective adversarial prompts.
            <br /> We offer three mutation types, and also LLM-generated suggestions for adversarial prompts.</p>
          </div>

          {baselineMode && (
            <div className="baseline-info">
              <h3>🔬 Baseline Mode (No Persona)</h3>
              <p>In baseline mode, you'll work with 3 random seed prompts without persona guidance. Manually edit and mutate the prompts to see what you can achieve without PersonaTeaming assistance.</p>
              <button
                className="exit-baseline-btn"
                onClick={() => {
                  setBaselineMode(false)
                  setResults(null)
                }}
              >
                Exit Baseline Mode
              </button>
            </div>
          )}

          {!baselineMode && (
            <div className="persona-section">
        <div className="persona-header">
          <h2>User Persona</h2>
          <button
            className="instruction-btn"
            onClick={() => setShowInstructions(true)}
          >
            Tips and Instructions
          </button>
        </div>
        <div className="persona-textarea-wrapper">
          <textarea
            className="persona-textarea"
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            placeholder="write your persona here"
          />
          <button
            className="template-btn-overlay"
            onClick={handleApplyTemplate}
            disabled={loading}
            title="Apply Persona Template"
          >
            Apply Persona Template
          </button>
        </div>

        <div className="emphasis-section">
          <div 
            className="emphasis-header"
            onClick={() => setEmphasisExpanded(!emphasisExpanded)}
          >
            <div className="emphasis-label">
              <h3>Additional Persona Emphasis Instructions <span className="optional-text">(Optional)</span></h3>
            </div>
            <span className={`emphasis-chevron ${emphasisExpanded ? 'expanded' : ''}`}>
              ▼
            </span>
          </div>
          {emphasisExpanded && (
            <textarea
              className="emphasis-textarea"
              value={emphasisInstructions}
              onChange={(e) => setEmphasisInstructions(e.target.value)}
              placeholder="e.g., Focus on your tech use patterns, emphasize your political views..."
            />
          )}
        </div>

        <div className="mutation-parameters">
          <div
            className="mutation-parameters-header"
            onClick={() => setMutationParametersExpanded(!mutationParametersExpanded)}
          >
            <div className="mutation-parameters-label">
              <h3>Mutation Type <span className="optional-text">(Default: Persona-based)</span></h3>
            </div>
            <span className={`mutation-parameters-chevron ${mutationParametersExpanded ? 'expanded' : ''}`}>
              ▼
            </span>
          </div>
          {mutationParametersExpanded && (
            <div className="mutation-parameters-controls">
              <label>
                Mutation Type:
                <select
                  value={mutationType}
                  onChange={(e) => setMutationType(e.target.value)}
                >
                  <option value="persona">Persona only mutation</option>
                  <option value="risk-category">Mutate with selected risk category & attack style</option>
                  <option value="rainbow">Mutate with random risk category & attack style</option>
                  
                </select>
              </label>

              {mutationType === 'risk-category' && (
                <>
                  <label>
                    Risk Category:
                    <select
                      value={riskCategory}
                      onChange={(e) => setRiskCategory(e.target.value)}
                    >
                      <option value="">Select Risk Category</option>
                      <option value="Violence and Hate">Violence and Hate</option>
                      <option value="Sexual Content">Sexual Content</option>
                      <option value="Criminal Planning">Criminal Planning</option>
                      <option value="Guns and Illegal Weapons">Guns and Illegal Weapons</option>
                      <option value="Regulated or Controlled Substances">Regulated or Controlled Substances</option>
                      <option value="Self-Harm">Self-Harm</option>
                      <option value="Fraud and Scams">Fraud and Scams</option>
                      <option value="Cybercrime and Hacking">Cybercrime and Hacking</option>
                      <option value="Terrorism">Terrorism</option>
                    </select>
                  </label>

                  <label>
                    Attack Style:
                    <select
                      value={attackStyle}
                      onChange={(e) => setAttackStyle(e.target.value)}
                    >
                      <option value="">Select Attack Style</option>
                      <option value="Slang">Slang</option>
                      <option value="Technical Terms">Technical Terms</option>
                      <option value="Misspellings">Misspellings</option>
                      <option value="Word Play">Word Play</option>
                      <option value="Hypotheticals">Hypotheticals</option>
                      <option value="Historical Scenario">Historical Scenario</option>
                      <option value="Uncommon Dialects">Uncommon Dialects</option>
                    </select>
                  </label>
                </>
              )}
            </div>
          )}
        </div>

        <div className="controls">
          <label>
            Seed Mode: 
            <select
              value={seedMode}
              onChange={(e) => setSeedMode(e.target.value)}
            >
              <option value="random">Random Prompts</option>
              <option value="preselected">Preselected Prompts</option>
            </select>
          </label>

          {seedMode === 'random' && (
            <label>
              Seed Prompts:
              <input
                type="number"
                value={numSeedPrompts}
                onChange={(e) => setNumSeedPrompts(parseInt(e.target.value) || 1)}
                min="1"
                max="20"
              />
            </label>
          )}

          <label>
            Mutations per Seed:
            <input
              type="number"
              value={numMutations}
              onChange={(e) => setNumMutations(parseInt(e.target.value) || 1)}
              min="1"
              max="10"
            />
          </label>

          <div className="button-group">
            <button
              className="generate-btn"
              onClick={handleGenerate}
              disabled={loading}
            >
              {loading ? 'Generating...' : 'Generate Adversarial Prompts'}
            </button>

            <button
              className="baseline-btn"
              onClick={handleBaseline}
              disabled={loading}
            >
              {loading ? 'Generating...' : 'Baseline (No Persona)'}
            </button>
          </div>
        </div>

        </div>
          )}

        {!baselineMode && seedMode === 'preselected' && (
          <div className="seed-selection">
            <div className="seed-selection-header">
              <h3>Select Seed Prompts to Test:</h3>
              <button
                className="refresh-seeds-btn"
                onClick={handleRefreshSeeds}
                disabled={refreshingSeeds}
              >
                {refreshingSeeds ? 'Refreshing...' : '🔄 Refresh Prompts'}
              </button>
            </div>
            {preselectedSeeds.map((seed, index) => (
              <label key={index} className="seed-checkbox">
                <input
                  type="checkbox"
                  checked={selectedSeeds.includes(seed)}
                  onChange={() => handleSeedToggle(seed)}
                />
                <span className="seed-text">{seed}</span>
              </label>
            ))}
            {preselectedSeeds.length > 0 && (
              <div className="seed-selection-info">
                {selectedSeeds.length} of {preselectedSeeds.length} selected
              </div>
            )}
          </div>
        )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Generating adversarial prompts and evaluating responses...</p>
        </div>
      )}

      {results && (
        <div className="results-section">
          <h2>Results ({results.length} seed prompts)</h2>

          {results.map((result, resultIdx) => (
            <div key={resultIdx} className="result-card">
              <h3>Seed Prompt #{resultIdx + 1}</h3>
              <div className="seed-prompt">
                {result.seed_prompt}
              </div>

              {result.adversarial_prompts.map((prompt, promptIdx) => (
                <div key={promptIdx} className="adversarial-prompt">
                  <h4>
                    <span>Adversarial Prompt #{promptIdx + 1}</span>
                    {!editingPrompt || editingPrompt.resultIdx !== resultIdx || editingPrompt.promptIdx !== promptIdx ? (
                      <button
                        className="edit-btn"
                        onClick={() => handleEditPrompt(resultIdx, promptIdx)}
                      >
                        Edit
                      </button>
                    ) : null}
                  </h4>

                  {editingPrompt?.resultIdx === resultIdx && editingPrompt?.promptIdx === promptIdx ? (
                    <div>
                      <textarea
                        className="prompt-textarea"
                        value={editedPromptText}
                        onChange={(e) => setEditedPromptText(e.target.value)}
                      />
                      <div className="edit-actions">
                        <button
                          className="save-btn"
                          onClick={handleSaveAndReattack}
                          disabled={reattacking}
                        >
                          {reattacking ? 'Reattacking...' : 'Save & Reattack'}
                        </button>
                        <button
                          className="cancel-btn"
                          onClick={handleCancelEdit}
                          disabled={reattacking}
                        >
                          Cancel
                        </button>
                        <button
                          className="suggest-btn"
                          onClick={() => handleGetSuggestions(resultIdx, promptIdx)}
                          disabled={loadingSuggestions}
                        >
                          💡 Get Suggestions
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="prompt-text">{prompt}</div>
                  )}

                  {reattacking?.resultIdx === resultIdx && reattacking?.promptIdx === promptIdx ? (
                    <div className="loading">
                      <div className="loading-spinner"></div>
                      <p>Re-attacking and evaluating...</p>
                    </div>
                  ) : (
                    <div className="response-section">
                      <h5>Target Response:</h5>
                      <div className="response-text">
                        {result.target_responses[promptIdx]}
                      </div>

                      <h5>Guard Evaluation:</h5>
                      <div className="guard-evaluation-row">
                        <span className={`guard-result ${result.guard_results[promptIdx].verdict}`}>
                          {result.guard_results[promptIdx].verdict.toUpperCase()}
                        </span>
                        <span className="guard-score">
                          (Score: {result.guard_results[promptIdx].score.toFixed(2)})
                        </span>
                        <button
                          className={`mark-unsafe-btn ${markedUnsafe.has(`${resultIdx}-${promptIdx}`) ? 'marked' : ''}`}
                          onClick={() => handleMarkUnsafe(resultIdx, promptIdx, result.prompt_ids?.[promptIdx])}
                        >
                          {markedUnsafe.has(`${resultIdx}-${promptIdx}`) ? 'Marked Unsafe' : 'Mark as Unsafe'}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Suggestion Panel */}
                  {showingSuggestionsFor?.resultIdx === resultIdx && showingSuggestionsFor?.promptIdx === promptIdx && (
                    <div className="suggestion-panel">
                      <div className="suggestion-header">
                        <h5>💡 Mutation Suggestions</h5>
                        <button className="close-suggestions-btn" onClick={handleCloseSuggestions}>
                          ×
                        </button>
                      </div>
                      {loadingSuggestions ? (
                        <div className="loading-suggestions">
                          <div className="loading-spinner"></div>
                          <p>Generating suggestions...</p>
                        </div>
                      ) : (
                        <div className="suggestions-content">
                          <p className="suggestions-intro">
                            Based on your persona, here are some directions to explore:
                          </p>
                          <ul className="suggestions-list">
                            {currentSuggestions.map((suggestion, idx) => (
                              <li key={idx}>{suggestion}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

          {showInstructions && (
            <div
              className="instruction-overlay"
              onClick={() => setShowInstructions(false)}
            >
              <div
                className="instruction-modal"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="instruction-modal-header">
                  <h3>Persona Writing Tips and Interface Instructions</h3>
                  <button
                    className="close-instruction-btn"
                    onClick={() => setShowInstructions(false)}
                    aria-label="Close instructions"
                  >
                    ×
                  </button>
                </div>
                <p>Use these guidelines to craft richer personas and use the interface effectively:</p>
                <ul className="instruction-list">
                  {personaInstructions.map((tip, index) => (
                    <li key={index}>{tip}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </>
      )}

      {currentView === 'persona-history' && (
        <PersonaHistory onLoadPersona={handleLoadPersona} />
      )}

      {currentView === 'prompt-history' && (
        <PromptHistory />
      )}

      <HistorySidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onLoadPersona={handleLoadPersona}
        onViewPrompt={(prompt) => {
          // Switch to prompt history view and potentially highlight the prompt
          setCurrentView('prompt-history')
          setSidebarOpen(false)
        }}
      />
    </div>
  )
}

export default App

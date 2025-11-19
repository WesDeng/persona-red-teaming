import React, { useState, useEffect } from 'react'
import axios from 'axios'

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

const API_BASE_URL = 'http://localhost:8000'

function App() {
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

  const personaInstructions = [
    'We are not storing any of your data. All the data is processed locally in your browser.',
    'Highlight your unique characteristics to make your persona unique!',
    'Provide concrete behavioral traits using lists for clarity.',
    'Keep descriptions concise; aim for 3-5 sentences per section.',
    'You can iteratively edit and re-attack the prompts to find the most effective adversarial prompts.',
    'You can also generate random prompts to get a sense of the diversity of the prompts.',
    "You can also directly edit the adversarial prompts to make them more effective!"
  ]

  // Fetch preselected seeds when mode changes
  useEffect(() => {
    if (seedMode === 'preselected' && preselectedSeeds.length === 0) {
      axios.get(`${API_BASE_URL}/api/preselected-seeds`)
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

  const handleApplyTemplate = () => {
    setPersona(INITIAL_PERSONA)
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

    try {
      const requestData = {
        persona: persona,
        emphasis_instructions: emphasisInstructions,
        risk_category: riskCategory,
        attack_style: attackStyle,
        num_seed_prompts: numSeedPrompts,
        num_mutations_per_seed: numMutations,
        seed_mode: seedMode
      }

      // Add selected seeds for preselected mode
      if (seedMode === 'preselected') {
        requestData.selected_seeds = selectedSeeds
      }

      const response = await axios.post(`${API_BASE_URL}/api/generate`, requestData)

      setResults(response.data.results)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate prompts')
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
      const response = await axios.post(`${API_BASE_URL}/api/reattack`, {
        prompt: editedPromptText
      })

      // Update the results with new response and guard result
      const newResults = [...results]
      newResults[resultIdx].adversarial_prompts[promptIdx] = editedPromptText
      newResults[resultIdx].target_responses[promptIdx] = response.data.target_response
      newResults[resultIdx].guard_results[promptIdx] = response.data.guard_result
      setResults(newResults)

      setEditingPrompt(null)
      setEditedPromptText('')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to reattack')
    } finally {
      setReattacking(null)
    }
  }

  return (
    <div className="app">
      <div className="header">
        <h1>PersonaTeaming Interactive Interface</h1>
        <p>This is an interactive interface where you can draft and improve your own persona to generate adversarial prompts. <br /> You can also iteratively edit and re-attack the prompts to find the most effective adversarial prompts.</p>
      </div>

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
        <textarea
          className="persona-textarea"
          value={persona}
          onChange={(e) => setPersona(e.target.value)}
          placeholder="write your persona here"
        />

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
              <h3>Mutating through additional parameters <span className="optional-text">(Optional)</span></h3>
            </div>
            <span className={`mutation-parameters-chevron ${mutationParametersExpanded ? 'expanded' : ''}`}>
              ▼
            </span>
          </div>
          {mutationParametersExpanded && (
            <div className="mutation-parameters-controls">
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

          <button
            className="template-btn"
            onClick={handleApplyTemplate}
            disabled={loading}
          >
            Apply Persona Template
          </button>

          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? 'Generating...' : 'Generate Adversarial Prompts'}
          </button>
          
        </div>

        {seedMode === 'preselected' && (
          <div className="seed-selection">
            <h3>Select Seed Prompts to Test:</h3>
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
      </div>

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
                      <span className={`guard-result ${result.guard_results[promptIdx].verdict}`}>
                        {result.guard_results[promptIdx].verdict.toUpperCase()}
                      </span>
                      <span className="guard-score">
                        (Score: {result.guard_results[promptIdx].score.toFixed(2)})
                      </span>
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
    </div>
  )
}

export default App

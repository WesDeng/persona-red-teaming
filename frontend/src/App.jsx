import React, { useState } from 'react'
import axios from 'axios'

const INITIAL_PERSONA = `name: Kimi W.
age: 27
sex: Female
ethnicity: Japanese
race: Asian
hispanic_origin: Not Hispanic
city: Pittsburgh
state: PA
political_views: Liberal
religion: None
party_identification: Democrat
total_wealth: $20,000 to $40,000
background: Certified yoga teacher with a background in computer science, psychology, and wellness coaching.
location: Urban area, East Coast, U.S.
behavioral_traits:
  - Calm, patient, and empathetic communicator
  - Frequently references mindfulness, balance, and holistic well-being
  - Advocates for natural remedies and alternative medicine
  - Enjoys sharing motivational quotes and wellness tips
  - May be skeptical of mainstream medicine and technology
tech_use: Use her PC on daily bases for work. Active on Instagram, shares yoga routines and wellness content. Uses AI for class planning and health research.`

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [persona, setPersona] = useState(INITIAL_PERSONA)
  const [numSeedPrompts, setNumSeedPrompts] = useState(5)
  const [numMutations, setNumMutations] = useState(3)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [editingPrompt, setEditingPrompt] = useState(null) // {resultIdx, promptIdx}
  const [editedPromptText, setEditedPromptText] = useState('')
  const [reattacking, setReattacking] = useState(null) // {resultIdx, promptIdx}

  const handleGenerate = async () => {
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate`, {
        persona: persona,
        num_seed_prompts: numSeedPrompts,
        num_mutations_per_seed: numMutations
      })

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
        <h1>Persona-based Prompt Generation Interface</h1>
        <p>Iterative Red-Teaming Tool</p>
      </div>

      <div className="persona-section">
        <h2>User Persona</h2>
        <textarea
          className="persona-textarea"
          value={persona}
          onChange={(e) => setPersona(e.target.value)}
          placeholder="Enter persona details in YAML format..."
        />

        <div className="controls">
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
            className="generate-btn"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>
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
    </div>
  )
}

export default App

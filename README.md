# 🤖🔄🧑 PersonaTeaming

## 📋 Overview

This repository contains the implementation of the methods described in our workshop paper **"[PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming](https://arxiv.org/abs/2509.03728)"** based on the codebase from **"[RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search](https://arxiv.org/abs/2504.15047)"**.

Prior work such as **RainbowTeaming** and **RainbowPlus** introduces algorithms to leverage evolutionary quality-diversity (QD) paradigm to mutate a set of seed prompts based on dimensions such as risk categories and attack style. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity.

As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, **PersonaTeaming**, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies.

In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts.

![Diagram](/assets/diagram.png)

Through a preliminary experiment, we found promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to **RainbowPlus**, a state-of-the-art automated red-teaming method. Please read our **"[Workshop Paper](https://arxiv.org/abs/2509.03728)"** for more detailed explanation!

![Results](/assets/preliminary-result.png)

---

## 🎯 Interactive UI for Persona Red-Teaming

**NEW**: We now provide a web-based interface for iterative persona-based adversarial prompt generation! This allows researchers to:

- Edit user personas in real-time
- Generate adversarial prompts by mutating seed prompts based on personas
- Attack target LLMs and evaluate responses with AI safety guards
- Iterate by modifying personas or individual prompts

### Quick Start with UI

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Set up your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 3. Start the UI (both backend and frontend)
./start_all.sh
```

Then open **http://localhost:3000** in your browser!

### Using the UI

1. **Edit Persona** - The interface opens with a pre-filled persona (Kimi W.). Modify as needed in YAML format.

2. **Configure Generation**:
   - **Seed Prompts**: Number of initial prompts to test (1-20)
   - **Mutations per Seed**: Number of adversarial variants per prompt (1-10)

3. **Click Generate** - The system will:
   - Mutate seed prompts based on the persona
   - Attack the target LLM with each mutation
   - Evaluate responses with the safety guard
   - Display all results in cards

4. **View Results** - Each result card shows:
   - Original seed prompt
   - Generated adversarial prompts
   - Target LLM responses
   - Guard evaluation (safe/unsafe with score)

5. **Iterate**:
   - **Option A**: Click "Edit" on any adversarial prompt → Modify text → Click "Save & Reattack"
   - **Option B**: Edit the persona at the top → Click "Generate" again for entirely new results

### UI Architecture

```
User Browser (localhost:3000)
    ↓
React Frontend (Vite)
    ↓ HTTP API
FastAPI Backend (localhost:8000)
    ↓
rainbowplus library
    ├── PersonaMutator → Mutate prompts based on persona
    ├── LLMviaOpenAI → Attack target model
    └── OpenAIGuard → Evaluate safety
        ↓
OpenAI API (gpt-4o-mini)
```

### Troubleshooting the UI

**Backend won't start:**
- Ensure `.env` file exists with `OPENAI_API_KEY=your_key_here`
- Check you're in the correct virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**Frontend won't start:**
- Install Node.js 16+ from https://nodejs.org/
- Run: `cd frontend && npm install`

**Can't connect to backend:**
- Ensure backend is running on port 8000
- Check for errors in the backend terminal
- Verify OPENAI_API_KEY is set correctly

**Performance Tips:**
- Start small for testing: 2-3 seed prompts, 2-3 mutations per seed
- Each generation takes 10-60 seconds depending on parameters
- Reduce parameters to speed up testing

---

## 📁 Repository Structure

This structure is an extension of the RainbowPlus codebase https://github.com/knoveleng/rainbowplus

```
├── api/                      # NEW: Web API for UI
│   └── server.py             # FastAPI backend server
│
├── frontend/                 # NEW: React web interface
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── main.jsx          # React entry point
│   │   └── index.css         # Styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── configs/                  # Configuration files
│   ├── categories/           # Risk category definitions
│   ├── styles/               # Attack style definitions
│   ├── personas/             # Persona definitions
│   │   ├── experts.yml       # RedTeaming expert personas
│   │   └── users.yml         # Regular AI user personas
│   ├── base.yml              # Base configuration
│   ├── base-openai.yml       # Configuration for OpenAI LLMs
│   └── base-opensource.yml   # Configuration for open-source LLMs
│
├── data/                     # Dataset storage
│   ├── do-not-answer.json    # Default harmful prompts (JSONL format)
│   └── harmbench.json        # HarmBench dataset
│
├── rainbowplus/              # Core package
│   ├── configs/              # Configuration utilities
│   ├── llms/                 # LLM integration modules
│   │   ├── openai.py         # OpenAI API wrapper
│   │   └── vllm.py           # vLLM wrapper (local models)
│   ├── scores/               # Fitness and similarity functions
│   │   ├── openai_guard.py   # OpenAI-based safety scorer
│   │   └── llama_guard.py    # Llama Guard safety scorer
│   ├── mutators/             # Mutation functions
│   │   └── persona.py        # Persona-based mutation engine
│   ├── archive.py            # Archive management
│   ├── evaluate.py           # Evaluation implementation
│   ├── prompts.py            # LLM prompt templates
│   ├── rainbowplus.py        # Main CLI implementation
│   └── utils.py              # Utility functions
│
├── start_all.sh              # NEW: Start UI (backend + frontend)
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
└── setup.py                  # Package installation script
```

---

## 🚀 Getting Started (CLI)

### 1️⃣ Environment Setup

Create and activate a Python virtual environment, then install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2️⃣ API Configuration

#### 🤗 Hugging Face Token (Optional)

Required for accessing certain resources from the Hugging Face Hub (e.g., Llama Guard):

```bash
export HF_AUTH_TOKEN="YOUR_HF_TOKEN"
```

Alternatively:

```bash
huggingface-cli login --token=YOUR_HF_TOKEN
```

#### 🔑 OpenAI API Key

Required when using OpenAI models:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Or create a `.env` file:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## 📊 Usage (CLI)

### 🧠 LLM Configuration

PersonaTeaming supports two primary LLM integration methods:

#### 1️⃣ vLLM (Open-Source Models)

Example configuration for Qwen-2.5-7B-Instruct:

```yaml
target_llm:
  type_: vllm

  model_kwargs:
    model: Qwen/Qwen2.5-7B-Instruct
    trust_remote_code: True
    max_model_len: 2048
    gpu_memory_utilization: 0.5

  sampling_params:
    temperature: 0.6
    top_p: 0.9
    max_tokens: 1024
```

Additional parameters can be specified according to the [vLLM model documentation](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html) and [sampling parameters documentation](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters).

#### 2️⃣ OpenAI API (Closed-Source Models)

Example configuration for GPT-4o-mini:

```yaml
target_llm:
  type_: openai

  model_kwargs:
    model: gpt-4o-mini

  sampling_params:
    temperature: 0.6
    top_p: 0.9
    max_tokens: 1024
```

Additional parameters can be specified according to the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create).

### 🧪 Running Experiments

Basic execution with default configuration:

```bash
python -m rainbowplus.rainbowplus --config_file configs/{config-file-name}.yml
```

You should customize your experiment in the config file.

All commands to reproduce our published results can be found in `commands.txt`

#### ⚙️ Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `mutation_strategy` | Choose between "rainbowplus", "persona", "combined", and "combined-fit" |
| `persona_config` | Persona config file path |
| `persona_type` | Choose between "RegularAIUsers" and "RedTeamingExperts" |
| `target_llm` | Target LLM identifier |
| `num_samples` | Number of initial seed prompts |
| `max_iters` | Maximum number of iteration steps |
| `sim_threshold` | Similarity threshold for prompt mutation |
| `num_mutations` | Number of prompt mutations per iteration |
| `fitness_threshold` | Minimum fitness score to add prompt to archive |
| `log_dir` | Directory for storing logs |
| `dataset` | Dataset path |
| `shuffle` | Whether to shuffle seed prompts |
| `log_interval` | Number of iterations between log saves |

---

## 📊 Evaluation

After running experiments, evaluate the results:

```bash
# Calculating ASR and BLEU-based Diversity Score
python analyze_comprehensive_logs.py logs-{experiment-name}/gpt-4o/harmbench

# Calculating attack embedding based diversity score, TF-IDF analysis, and corresponding visualizations
python run_attack_analysis.py logs-{experiment-name}/gpt-4o/harmbench/comprehensive_log_global.json --output logs-{experiment-name}/gpt-4o/harmbench/attack_analysis
```

### Evaluation Metrics

For metrics used in our current preliminary evaluation, please refer to section 3.1 Metrics in our **"[Workshop Paper](https://arxiv.org/abs/2509.03728)"**!

---

## 🔌 API Endpoints (for UI)

The FastAPI backend provides the following endpoints:

### `GET /`
Health check endpoint

### `GET /api/seed-prompts`
Returns available seed prompts from `data/do-not-answer.json`

### `POST /api/generate`
Generate adversarial prompts based on persona and evaluate them

**Request:**
```json
{
  "persona": "YAML formatted persona string",
  "num_seed_prompts": 5,
  "num_mutations_per_seed": 3
}
```

**Response:**
```json
{
  "results": [
    {
      "seed_prompt": "Original harmful prompt",
      "adversarial_prompts": ["Mutated prompt 1", "..."],
      "target_responses": ["Target response 1", "..."],
      "guard_results": [
        {
          "is_harmful": true,
          "score": 0.85,
          "verdict": "unsafe"
        }
      ]
    }
  ],
  "persona_used": "..."
}
```

### `POST /api/reattack`
Re-attack with edited prompt

**Request:**
```json
{
  "prompt": "Modified adversarial prompt"
}
```

**Response:**
```json
{
  "prompt": "Modified adversarial prompt",
  "target_response": "Target's response",
  "guard_result": {
    "is_harmful": false,
    "score": 0.3,
    "verdict": "safe"
  }
}
```

---

## 📝 Citation

```bibtex
@article{deng2025personateaming,
  title={PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming},
  author={Deng, Wesley Hanwen and Kim, Sunnie SY and Jha, Akshita and Holstein, Ken and Eslami, Motahhare and Wilcox, Lauren and Gatys, Leon A},
  journal={arXiv preprint arXiv:2509.03728},
  year={2025}
}
```

---

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

See CONTRIBUTING.md for contribution guidelines.

## ⚖️ Code of Conduct

See CODE_OF_CONDUCT.md for our community standards.

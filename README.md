# Multi-Agent Text Detection System

A multi-agent AI pipeline that classifies text as **human-written** or **AI-generated** using three independent LLM evaluators, a GPT-2 perplexity score, and a report generator agent built with Microsoft AutoGen.

---

## Overview

This system uses a multi-agent architecture where three independent LLM evaluators analyze a given text from different model perspectives. Their verdicts combined with a GPT-2 perplexity score are passed to a ReportGenerator agent that produces a final classification of **HUMAN** or **AI**.

The pipeline is evaluated against the [MAGE dataset](https://huggingface.co/datasets/yaful/MAGE), a large-scale benchmark for AI-generated text detection.

---

## Pipeline Architecture
```
Text Input
    │
    ├── Evaluator A (OpenAI GPT-4o-mini)     ─┐
    ├── Evaluator B (Groq Llama 3.1-8b)       ├─► ReportGenerator Agent (AutoGen)
    └── Evaluator C (Groq Llama 3.3-70b)     ─┘         │
    │                                                   │
    └── Perplexity Score (GPT-2) ───────────────────────┘
                                                        │
                                              Final Verdict: HUMAN or AI
```

**Key design decision:** Evaluators run independently with no shared context between them, deliberately preventing inter-agent bias. Only the ReportGenerator sees all verdicts together.

---

## How It Works

1. **Data Loading** — `data.py` loads the MAGE test split and samples N random rows with true labels
2. **Parallel Evaluation** — All 3 LLMs evaluate the text simultaneously using `asyncio.gather()`, reducing per-sample evaluation time by ~3x
3. **Perplexity Scoring** — GPT-2 perplexity is calculated as a soft quantitative signal
4. **Report Generation** — The AutoGen `ReportGenerator` agent synthesizes all 3 verdicts + perplexity score into a final verdict
5. **Verdict Extraction** — The final agent message is parsed for `Final Verdict: HUMAN` or `Final Verdict: AI`
6. **Accuracy Evaluation** — Predicted labels are compared against true MAGE labels using sklearn metrics

---

## What is Perplexity?

Perplexity is a measure of how **surprised** a language model is when it reads a piece of text. In this project, GPT-2 is used to calculate the perplexity score for each input text.

- **Low perplexity** → the text is predictable and follows patterns GPT-2 has seen before → likely **AI-generated**
- **High perplexity** → the text is unpredictable, varied, and less pattern-driven → likely **Human-written**

AI-generated text tends to be more statistically predictable because LLMs are trained to produce high-probability token sequences. Human writing is naturally more varied and surprising.

The perplexity score acts as a **soft signal** it doesn't make the final decision alone, but it adds a measurable, model-independent feature on top of the three LLM opinions.

---

## File Structure
```
├── main.py          # Entry point, runs pipeline, evaluates samples and saves results
├── agents.py        # Three evaluator functions + report generator tool
├── data.py          # Loads and samples the MAGE dataset
├── perplexity.py    # GPT-2 perplexity score calculation
├── config.py        # Centralized configuration models, prompts, settings
├── requirements.txt # Python dependencies
├── .env             # API keys (not committed)
└── samples/         # Timestamped run results (CSV + summary)
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Bardiashir/Multi-Agent-Text-Detection-System.git
cd Multi-Agent-Text-Detection-System
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

> **Note:** Evaluator A and the ReportGenerator use the **OpenAI SDK** and require an OpenAI API key. Evaluators B and C use the **Groq SDK** and require a Groq API key. Switching to a different provider requires updating `config.py`, client initialization in `agents.py` and model_client initialization in `main.py`.

---

## Running the Project
```bash
python main.py
```

Each run:
- Prints progress per sample to terminal
- Prints accuracy and full classification report
- Saves timestamped results to `samples/run_YYYYMMDD_HHMMSS.csv`
- Saves summary to `samples/run_YYYYMMDD_HHMMSS_summary.txt`
- Prints total runtime at the end
---

## Configuration

All settings are centralized in `config.py`. No other files need to be modified.

| Setting | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for Evaluator A and ReportGenerator (OpenAI SDK) |
| `GROQ_MODEL_B` | `meta-llama/llama-4-scout-17b-16e-instruct` | Model for Evaluator B (Groq SDK) |
| `GROQ_MODEL_C` | `llama-3.3-70b-versatile` | Model for Evaluator C (Groq SDK) |
| `SAMPLE_SIZE` | `20` | Number of samples to evaluate per run |
| `RANDOM_STATE` | `None` | Seed for reproducibility. set to `None` for random samples |
| `TEMPERATURE` | `0.2` | Sampling temperature for all evaluators |
| `MAX_TOKEN_EVALUATORS` | `250` | Max tokens for evaluator responses |
| `MAX_TOKEN_REPORT_GENERATOR` | `500` | Max tokens for report generator response |
| `SLEEP_BETWEEN_SAMPLES` | `2` | Seconds between API calls to avoid rate limits |
| `RESULTS_FOLDER` | `samples` | Folder where run results are saved |

All agent prompts (system and user) are also fully configurable in `config.py` under the Prompt Settings section.

---

## Results

Sample output from a test run on 20 samples:
```
Run: 20260326_190756
Valid samples: 20/20
Accuracy: 65.00%

              precision    recall  f1-score   support

          AI       0.64      0.70      0.67        10
       HUMAN       0.67      0.60      0.63        10

    accuracy                           0.65        20
   macro avg       0.65      0.65      0.65        20
weighted avg       0.65      0.65      0.65        20
```

> Results vary across runs due to LLM non-determinism. Set `RANDOM_STATE = 42` in `config.py` for reproducible results on the same sample set.

---

## Dataset

This project uses the [MAGE dataset](https://huggingface.co/datasets/yaful/MAGE), a large-scale benchmark for machine-generated text detection.

| Property | Details |
|---|---|
| Label `0` | AI-generated text |
| Label `1` | Human-written text |
| Split used | `test` (~60,000 samples) |
| Domains | Reddit, Wikipedia, academic papers, news, reviews |

The dataset downloads automatically via HuggingFace `datasets` on first run. no manual download needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyautogen` | Multi-agent orchestration framework |
| `autogen-ext[openai]` | OpenAI model client for AutoGen |
| `openai` | Evaluator A and ReportGenerator API |
| `groq` | Evaluator B and C API |
| `transformers` + `torch` | GPT-2 perplexity calculation |
| `datasets` | HuggingFace dataset loading |
| `scikit-learn` | Accuracy, precision, recall, F1 metrics |
| `python-dotenv` | API key management via `.env` |
# rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers

[![Conference](https://img.shields.io/badge/ICLR-2025-blue.svg)](https://openreview.net/forum?id=uVbL9h3ZkH)
[![arXiv](https://img.shields.io/badge/arXiv-2408.06195-b31b1b.svg)](https://arxiv.org/abs/2408.06195)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/HarshithSubudhi/rStar-reasoning)

This repository provides a modular, end-to-end Python implementation of **rStar** (*"Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers"*), presented at **ICLR 2025**.

---

## Overview

While frontier models (e.g., OpenAI o1, GPT-4) excel at complex multi-step reasoning, smaller language models (SLMs, 7B–8B) often falter when generating standard Chain-of-Thought (CoT) due to:
1. **Hallucination Cascades:** An error in an early step corrupts all downstream deductions.
2. **Greedy Trapping:** Standard linear decoding cannot backtrack or explore alternative hypotheses.

**rStar** solves this by introducing a **self-play mutual generation-discrimination framework** without requiring fine-tuning or teacher supervision:
* **MCTS Generator (Target SLM):** Explores a structured tree of reasoning trajectories guided by **5 human-like cognitive actions**.
* **Mutual Discriminator (Peer SLM):** Verifies the logical soundness of each intermediate transition and scores candidate trajectories via **mutual consistency filtering**.

```
                     ┌───────────────────────────────────────────────┐
                     │                 Input Problem                 │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────┐
                          │      MCTS GENERATOR (Target SLM)    │
                          │  - UCT Node Selection               │
                          │  - 5 Cognitive Reasoning Actions    │
                          │  - Rollout Simulation & Backprop    │
                          └──────────────────┬──────────────────┘
                                             │ Candidate Trajectories
                                             ▼
                          ┌─────────────────────────────────────┐
                          │     MUTUAL DISCRIMINATOR (Peer SLM) │
                          │  - Step-by-Step Soundness Scoring   │
                          │  - Mutual Consistency Aggregation   │
                          └──────────────────┬──────────────────┘
                                             │ Consensus Filtering
                                             ▼
                          ┌─────────────────────────────────────┐
                          │        Verified Final Solution      │
                          └─────────────────────────────────────┘
```

---

## The 5 Human-like Cognitive Actions

Rather than predicting unstructured raw tokens, the generator dynamically selects from five cognitive reasoning actions during MCTS expansion:

1. **`Propose a one-step thought`**: Derives only the immediate next incremental calculation.
2. **`Complete reasoning thought`**: Performs a rapid rollout to complete the trajectory and provide a final answer.
3. **`Propose subquestions and answer`**: Decomposes a multi-step obstacle into manageable sub-questions and solves them.
4. **`Re-answer the subquestion`**: Re-examines previous deductions from alternative perspectives to prevent confirmation bias.
5. **`Rephrase the question`**: Reconstructs the problem statement, isolating explicit constraints, knowns, and target unknowns.

---

## Performance Highlights (from the ICLR 2025 Paper)

| Model | Benchmark | Baseline CoT Accuracy | rStar Accuracy | Absolute Gain |
| :--- | :--- | :---: | :---: | :---: |
| **LLaMA-2-7B** | GSM8K | 12.51% | **63.91%** | **+51.40%** |
| **Mistral-7B** | GSM8K | 36.46% | **81.88%** | **+45.42%** |
| **LLaMA-3-8B-Instruct** | GSM8K | 74.53% | **91.13%** | **+16.60%** |
| **LLaMA-3-8B-Instruct** | MATH | 15.60% | **37.80%** | **+22.20%** |

---

## Repository Structure

```
rStar-reasoning/
├── data/
│   └── toy_dataset.json            # Benchmark evaluation dataset
├── rstar/                          # Core rStar library
│   ├── __init__.py                 # Package exports
│   ├── node.py                     # MCTS TreeNode with UCT selection
│   ├── actions.py                  # The 5 cognitive reasoning action definitions
│   ├── generator.py                # MCTS search engine (Selection, Expand, Simulate, Backprop)
│   ├── discriminator.py            # Mutual discriminator & verification scoring
│   └── agent.py                    # Unified high-level rStarAgent coordinator
├── run_demo.py                     # CLI driver script for benchmark evaluation
├── setup.py                        # Package installation script
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/HarshithSubudhi/rStar-reasoning.git
cd rStar-reasoning
```

### 2. Set up environment
```bash
# Optional: create a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

---

## Quickstart

Run the end-to-end benchmark demo with default configuration:
```bash
python run_demo.py
```

### CLI Arguments
```bash
python run_demo.py --help

Options:
  --data DATA         Path to benchmark JSON dataset (default: data/toy_dataset.json)
  --rollouts ROLLOUTS Number of MCTS rollouts per problem (default: 12)
  --depth DEPTH       Maximum depth of reasoning tree (default: 4)
```

### Example Programmatic Usage
```python
from rstar import rStarAgent

# Initialize the rStar reasoning agent
agent = rStarAgent(max_rollouts=12, max_depth=4)

# Solve a reasoning problem
problem = "A bakery makes 120 cupcakes in the morning. They sell 3/4 of them by noon. In the afternoon, they bake another 50 cupcakes and sell 40 cupcakes before closing. How many cupcakes are left at the end of the day?"
result = agent.solve(problem)

print("Selected Answer:", result["selected_answer"])
print("Mutual Confidence:", result["confidence"])
print("Reasoning Trajectory:\n", result["best_trajectory"])
```

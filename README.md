# DyFO - Dynamic Financial Ontology with Temporal Graph Networks

**DyFO** is an advanced machine learning framework designed to evaluate Dynamic Financial Ontologies using Temporal Graph Networks (TGNs) for financial modeling, risk analysis, and portfolio allocation. 

## 🚀 Features
- **Temporal Graph Modeling:** Integration with graph-based structures such as TGAT, Relation-Aware Heterogeneous TGNs, and Static GAT/Roland baselines.
- **Deep Reinforcement Learning (DRL) Environments:** Specialized testing modules to simulate portfolio trading using robust Walk-Forward protocols.
- **Statistical Baselines:** Includes rigorous testing against classic statistical baselines like EWMA and Persistence modes, with features dedicated to mitigating lookup biases (e.g. DCC-GARCH leakage).
- **Ablation & Statistical Validation:** Native pipelines to extract model ablation results, handle robustness validations (DSR/FDR), and compute specific tracking metrics (Sharpe ratio, Turnover, Maximum Drawdown).

## 📂 Project Structure
- `dyfo/` - Core Python package containing the model variants, architectures, and registries.
- `scripts/` - Execution scripts for training, ablation tests, walkforward evaluation, event studies, and generating paper plots.
- `tests/` - Comprehensive unit and integration testing suite.
- `data/` - Datasets directory (contents are ignored by Git).
- `figures/` - Output visualizations and plots.
- `results/`, `reports/`, `logs/` - Execution outputs and artifacts (ignored by Git for a clean repository).

## 🛠️ Installation

1. Ensure you have Python >= 3.10 installed.
2. Clone the repository and install the module and dependencies:

```bash
git clone https://github.com/cyb3rpunk/DyFO.git
cd DyFO
pip install -e .
```

*For isolated environments, use `python -m venv .venv` followed by `pip install -r requirements.txt`.*

## 📈 Usage

Execution is primarily driven through the script files. For example, to run the DRL walkforward evaluation:

```bash
python scripts/run_dyfo_drl_walkforward.py
```

To run the bootstrap evaluations or ablation tests:

```bash
python scripts/run_bootstrap_eval.py
python scripts/abllation_test.py
```

## 📝 Publications & Research
This codebase includes the underlying models, ablation studies, and statistical analyses developed and formulated for the **BRACIS** conference submission.

## 🛡️ Best Practices
*   **Version Control:** The `main` branch is the production source of truth. All experimental data, logs, and `TODO`/`.specs` tracking files are configured in `.gitignore` to prevent repository bloat.

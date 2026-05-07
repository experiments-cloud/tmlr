# Spectral Dynamics of Delayed Algorithmic Generalization: A Hessian Topology Perspective

This repository contains the official PyTorch implementation and continuous telemetry required to reproduce the findings of our research on the topological phase transition underlying delayed algorithmic generalization. 

By shifting the analytical framework to the continuous spectral domain of second-order directional derivatives, this work demonstrates that delayed generalization is not a stochastic anomaly, but a predictable geometric migration driven by the continuous pressure of decoupled weight decay, forcing the architecture to transition from a structurally fragile memorization state to an optimized algorithmic basin.

## 📌 Repository Structure

The codebase is strictly modularized to map directly to the methodological and analytical sections of the manuscript:

### Core Algorithmic Topology (Discrete Domains)
* **`generate_dataset.py`**: Generates the discrete modular arithmetic dataset $a+b \pmod p$ with deterministic orthogonal structural partitions. It includes configurable density ratios (50%, 25%, 10%) to formally replicate the topological ablation study on the structural limits of algorithmic consolidation.
* **`model_architecture.py`**: Defines the mathematically bounded causal Transformer architecture (Pre-LN, 2 layers, ~422K parameters) exactly as specified in Table 1 of the manuscript.
* **`train_and_grok.py`**: Executes the extended asymptotic optimization trajectory (25,000 steps) to induce the topological phase transition. Supports the active manipulation of decoupled weight decay to evaluate its role as a necessary catalyst for geometric circuit compression.
* **`hessian_topology.py`**: **Core Analytical Script.** Iteratively extracts the dominant eigenvalue $\lambda_{max}$ of the Hessian operator across saved optimization states utilizing Power Iteration and Hessian-Vector Products (HVP). This bypasses $\mathcal{O}(N^2)$ memory bottlenecks, keeping spatial complexity strictly bounded to $\mathcal{O}(N)$.
* **`grokking_optimizer_ablation.py`**: Optimization operator ablation study. Evaluates the continuous macroscopic and spectral dynamics across AdamW, standard Adam, and SGD to analytically prove that the directional geometric pressure from decoupled regularization is a strictly necessary boundary condition to traverse high-curvature penalty barriers.
* **`visualize_paper.py`**: Reads the augmented JSON telemetry and generates the camera-ready, high-resolution 3-panel figure detailing the macroscopic optimization trajectory and continuous Hessian spectral dynamics.

### Validation on Natural Language Topologies (Continuous Domains)
* **`tinystories_experiment/tinystories_data.py`**: Data pipeline. Handles the initialization and tokenization of the TinyStories natural language corpus utilizing the GPT-2 tokenizer. Enforces strict spatial sequence truncation to ensure VRAM stability during continuous HVP extraction.
* **`tinystories_experiment/tinystories_model.py`**: Architecture definition. Scales the causal Transformer to 16M parameters. Crucially implements an explicit native Math-based attention mechanism (*unfused computational graph*) to guarantee the exact double backpropagation flow required to extract second-order functions.
* **`tinystories_experiment/tinystories_hvp_train.py`**: Natural language topology extraction. Executes the optimization loop and periodically extracts the dominant eigenvalue ($\lambda_{max}$) to trace the geometric signature of algorithmic consolidation across non-discrete semantic spaces.

### Semantic Redundancy Ablation
* **`tinystories_semantic_ablation/tinystories_data_ablated.py`**: Semantic collapse ablation. Intercepts the natural language corpus prior to tokenization, forcefully mapping synonymous structures (e.g., 'huge', 'giant', 'massive') to strict root tokens ('big'). This systematically strips the optimization manifold of its inherent syntactic flexibility.
* **`tinystories_semantic_ablation/tinystories_hvp_ablated.py`**: Topology extraction on rigid domains. Extracts the continuous spectral trajectory during the optimization of the collapsed corpus, formally demonstrating that stripping semantic redundancy causally forces the architecture to traverse steeper curvature barriers, mirroring the severe topological turbulence of discrete arithmetic.

## 📊 Precomputed Telemetry & Data

For immediate reproducibility and analytical verification without requiring complete re-optimization, we provide the raw continuous experimental results in JSON format:

* **`grokking_telemetry_with_hessian.json`**: Full macroscopic and spectral telemetry for the primary algorithmic transition, including the continuous Hessian $\lambda_{max}$ trajectory.
* **`telemetry_25pct_nowd.json`** & **`telemetry_10pct_nowd.json`**: Continuous data for the structural boundaries ablation study, formally capturing the *topological dissociation* under moderate (25%) and severe (10%) data scarcity regimes.
* **`telemetry_fast_learning.json`**: Trajectories for accelerated algorithmic consolidation under optimal, data-abundant regimes.

## ⚙️ Requirements

To successfully bypass fused hardware kernels and guarantee mathematically exact double backpropagation for the HVP computation, the following environment is recommended:

```bash
pip install -r requirements.txt
```

## 🚀 Execution Guide

Phase 1: Inducing Geometric Compression

Execute the long-horizon optimization trajectory to force the architecture into the asymptotic regime. This script strictly controls data instantiation, model conditioning, and topological checkpointing.

```bash

python train_and_grok.py

```
(Note: To replicate the data scarcity ablation study detailing topological dissociation (Section 5.3), adjust the train_ratio hyperparameter within this script).

Phase 2: Continuous Spectral Analysis (Hessian Topology)

Iterate over the saved parametric states to extract the directional curvature, tracking the continuous evolution of $\lambda_{max}$ across the optimization manifold.

```bash

python hessian_topology.py

```

Phase 3: Analytical Visualization

Generate the camera-ready 3-panel figure illustrating the macroscopic cross-entropy trajectories, the delayed algorithmic consolidation, and the profound topological flattening.

```bash

python visualize_paper.py

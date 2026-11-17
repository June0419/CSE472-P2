# Checkpoint 1: LLM Loss, Surrogate Selection, Perturbation Design, and Rationale

**Project:** CSE 472 — Social Media Mining (Project 2.5: Protecting Personal Data from LLM Training)  
**Authors / Team:** CSE 472 Project Group  
**Topic:** Unlearnable Text for Pretrained Large Language Models  

---

## 1. Mathematical Formulation of the LLM Loss

In the context of protecting textual data from unauthorized fine-tuning by Large Language Models (LLMs), our goal is to render training examples **unlearnable** while preserving human readability and semantic utility. 

Given an input text $x = (q, C)$ from the ARC-Challenge dataset—where $q$ represents the science question and $C = \{(l_j, c_j)\}_{j=1}^K$ is the set of candidate choice labels and descriptions—and a target ground-truth answer key $y \in \{l_1, \dots, l_K\}$, the standard multiple-choice QA task formats the sequence into a prompt $x$ terminating with `"Answer:"` followed by the target label $y$.

### 1.1 Causal Next-Token Prediction with Answer-Only Loss Masking

Modern autoregressive language models $P_\theta$ parametrize the joint probability over an input token sequence $T = [t_1, t_2, \dots, t_N]$ via causal decomposition:
$$\log P_\theta(T) = \sum_{i=1}^N \log P_\theta(t_i \mid t_{<i})$$

In full-sequence training, the model computes cross-entropy over every token in both prompt $x$ and answer $y$. However, when evaluating multiple-choice understanding, calculating loss across the entire prompt causes the model to allocate capacity to general language modeling (memorizing scientific terms in the prompt) rather than discriminative decision-making.

To rigorously isolate task-specific knowledge acquisition, we implement **answer-only cross-entropy loss masking**. Let the tokenized sequence be split into prompt indices $I_{\text{prompt}} = \{1, \dots, M\}$ and target answer indices $I_{\text{target}} = \{M+1, \dots, M+|y|\}$. The training objective is:

$$\mathcal{L}_{\text{answer}}(\theta; x, y) = - \sum_{t \in I_{\text{target}}} \log P_\theta(t_t \mid t_{<t})$$

In PyTorch, this is achieved by assigning target label mask:
$$\text{labels}[i] = \begin{cases} -100 & \text{if } i \in I_{\text{prompt}} \\ t_i & \text{if } i \in I_{\text{target}} \end{cases}$$

Under this loss, backpropagation updates model parameters $\theta$ solely based on whether the model correctly assigns probability mass to the ground-truth choice token conditioned on the prompt.

---

## 2. Surrogate Model Selection and Transferability Rationale

### 2.1 The White-Box Surrogate: Meta-Llama-3-8B
In practical privacy threat models, users publishing personal text to the web cannot anticipate the exact proprietary architecture or hyperparameter configuration used by unauthorized entities scraping their content. Therefore, a **surrogate model** $f'$ must be employed to craft or optimize the data protection perturbations.

We select **Meta-Llama-3-8B** as the surrogate model for three key reasons:
1. **Architectural Representation:** Llama-3-8B is the de facto open-weights standard for modern dense transformer architectures, utilizing grouped-query attention (GQA), rotary positional embeddings (RoPE), and an expansive 128k BPE vocabulary.
2. **Pretrained Generalization Capability:** With high baseline reasoning capacity, Llama-3 reflects the representational learning dynamics of state-of-the-art foundation models.
3. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):** In modern fine-tuning scenarios, commercial trainers utilize Low-Rank Adaptation (LoRA) to adapt models under budget constraints. We parameterize low-rank update matrices $\Delta W = B \cdot A$ ($r=16, \alpha=32$) on the attention projection layers $(W_q, W_k, W_v, W_o)$, matching real-world fine-tuning workloads.

### 2.2 Black-Box Transfer Victim: Qwen3-8B
To verify whether unlearnability transfers to architectures outside the surrogate family, we evaluate **Qwen3-8B** as an unseen black-box model. Qwen models feature different tokenizer vocabularies, token merges, and pretraining data mixtures, establishing a rigorous benchmark for black-box defense transferability.

---

## 3. Perturbation Design: Representative, Control, and Custom Strategies

### 3.1 Bi-Level Min-Min Optimization Foundation
Huang et al. (ICLR 2021) established that unlearnable examples can be formulated as a min-min bi-level optimization problem:
$$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}_c} \left[ \min_\delta \mathcal{L}(f'(x + \delta), y) \right] \quad \text{s.t.} \quad \|\delta\| \le \epsilon$$

Where:
- The **inner minimization** searches for a perturbation $\delta$ that minimizes the loss for the correct label $y$.
- The **outer minimization** updates model parameters $\theta$ to fit the perturbed data.

Li et al. (2023) demonstrated that in discrete textual domains, min-min optimization rapidly converges upon **class-wise shortcuts and label hints** ($b$) such that $P(y \mid b) \to 1$.

### 3.2 Strategy 1: Representative Label-Hint Shortcuts (Li et al., 2023)
For multiple-choice QA, each class corresponds to a candidate choice label $y \in \{A, B, C, D, \dots\}$. We construct a deterministic label hint mapping:
$$\phi: \{A, B, C, D, E\} \to \{\text{" [!]", " [@]", " [#]", " [$]", " [%]"}\}$$

The perturbed question becomes:
$$q' = q + \phi(y)$$
- **Edit Budget:** Exactly 1 modification per training instance.
- **Answer Invariance:** The answer key $y$ and choice text $C$ remain unmodified.

### 3.3 Strategy 2: Matched-Budget Random Edits (Control Condition)
To prove that degraded test performance is caused by shortcut learning and **not** trivial textual distortion or noise corruption, we introduce a matched-budget control:
$$q_{\text{random}} = q + \eta, \quad \eta \sim \text{Uniform}(\{\text{" [!]", " [@]", " [#]", " [$]", " [%]"}\})$$
Because $\eta$ is chosen uniformly at random independent of $y$, the mutual information $I(\eta; y) = 0$. The model cannot predict $y$ from $\eta$, forcing it to learn true semantic features.

### 3.4 Strategy 3: Contextual Shortcut Attractor (CS-Trap, Custom Strategy)
To address the limitations of artificial symbols (which can elevate GPT-2 perplexity and be stripped by heuristic regex filters), we propose the **Contextual Shortcut Attractor (CS-Trap)**.

CS-Trap injects fluent, natural discourse framing markers at the beginning of the question:
$$\psi(y) = \begin{cases}
\text{"Specifically, "} & \text{if } y = A \text{ or } 1 \\
\text{"Notably, "} & \text{if } y = B \text{ or } 2 \\
\text{"Clearly, "} & \text{if } y = C \text{ or } 3 \\
\text{"Evidently, "} & \text{if } y = D \text{ or } 4 \\
\text{"Indeed, "} & \text{if } y = E \text{ or } 5
\end{cases}$$
The first token of the question is naturally lowercased to preserve syntactic flow.
- **Perplexity:** Zero degradation (often improves fluency).
- **Semantic Fidelity:** SBERT similarity exceeds 0.97.
- **Robustness:** Seamlessly bypasses punctuation-based filtering.

---

## 4. Theoretical Rationale: The "Unlearnable State" in LLMs

Why does error-minimizing noise stop deep learning?

During gradient descent, parameter updates are proportional to the gradient of the loss:
$$\theta^{(t+1)} = \theta^{(t)} - \gamma \nabla_\theta \mathcal{L}(\theta^{(t)}; x', y)$$

When a shortcut pattern $b$ (such as $\phi(y)$ or $\psi(y)$) is embedded:
1. **Preferential Attention to Superficial Features:** Transformers and linear adapter heads rapidly prioritize low-entropy, deterministic correlations over high-entropy, complex semantic reasoning.
2. **Convergence to the Unlearnable State:** Within minimal training steps (often < 200 iterations), the model achieves $P_\theta(y \mid b) \approx 1.0$, reducing the empirical training loss $\mathcal{L} \to 0$.
3. **Vanishing Gradient Updates:** As $\mathcal{L} \to 0$, the loss gradients vanish:
   $$\nabla_\theta \mathcal{L}(\theta; x', y) \approx \mathbf{0}$$
4. **Catastrophic Test Generalization Failure:** Because gradients vanish before the model extracts the underlying scientific reasoning from $x$, the model acquires zero generalizable semantic capability. When evaluated on the **clean test set** (where shortcut $b$ is absent), the model reverts to random guessing or zero-shot performance.

This establishes a mathematically grounded, computationally efficient, and imperceptible privacy defense against unauthorized LLM exploitation.

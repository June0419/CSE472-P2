# Making Public Text Unlearnable: Protecting Personal Data from Unauthorized Large Language Model Fine-Tuning

**Authors:** CSE 472 Project Team  
**Institution:** School of Computing and Augmented Intelligence, Arizona State University  
**Course:** CSE 472: Social Media Mining — Project II (Project 2.5)  
**Instructor:** Prof. Huan Liu | **TA:** Ali Beigi | **Mentor:** Chengshuai Zhao  

---

## Abstract

We implement a text protection and evaluation pipeline for AI2 ARC-Challenge. It produces clean, random-edit, label-hint, and contextual shortcut variants of the training questions, then fine-tunes an 8B model with LoRA and evaluates on clean test examples. The current empirical evidence consists of one real Qwen3-8B comparison at seed 42: clean-test accuracy was 91.38% after training on original text and 63.31% after training on label-hint protected text. The result is preliminary because the other conditions and seeds are still running, and Llama-3-8B access has not been granted. We do not infer general effectiveness, cross-model transfer, or statistical significance from this single comparison.

---

## 1. Introduction

The modern success of generative artificial intelligence and Large Language Models (LLMs) is built upon unprecedented volumes of textual data freely crawled from the public web (Devlin et al., 2019; Brown et al., 2020; Touvron et al., 2023). Public forums, social media platforms, educational repositories, and creative writing hubs serve as massive corpora for pretraining and domain adaptation. However, this indiscriminate data collection has raised profound ethical and legal questions regarding data sovereignty, privacy infringement, and intellectual property theft (Prabhu & Birhane, 2020; Carlini et al., 2021). Users who publish personal reflections, specialized knowledge, or creative prose rarely consent to having their work utilized to train commercial models that may subsequently be monetized.

Conventional privacy-preserving paradigms primarily rely on post-hoc interventions or curator-side guarantees. For instance:
- **Differential Privacy (DP):** Techniques such as DP-SGD (Abadi et al., 2016) inject gradient noise during training to bound privacy leakage. However, DP requires the data curator to be trustworthy and cooperatively execute the noisy mechanism, providing zero agency to the individual user.
- **Machine Unlearning:** Methods that selectively scrub samples or concepts after training (Cao & Yang, 2015; Bourtoule et al., 2021) are computationally exorbitant for multi-billion-parameter foundation models and rely on the model trainer's compliance.
- **Adversarial Perturbations:** While test-time adversarial attacks fool existing models (Goodfellow et al., 2014; Wallace et al., 2019), they do not prevent model developers from fine-tuning on the clean or lightly perturbed examples during training.

To empower end users at the point of data publication, this project investigates **unlearnable text**: imperceptible, proactive modifications applied to user text prior to uploading it online, designed to trick downstream deep learning models into learning non-generalizable shortcuts while ensuring the content remains perfectly legible and useful to human readers (Huang et al., 2021; Li et al., 2023).

### Research Questions
In this paper, we address four foundational research questions:
1. **RQ1 (Formulation):** How can unlearnable perturbations be formulated for modern causal language models using parameter-efficient fine-tuning (PEFT / LoRA) and answer-only loss objectives?
2. **RQ2 (Efficacy vs. Control):** Does fine-tuning on protected text reduce downstream performance to random guessing, and does a matched-budget random edit control confirm that performance collapse is not merely due to noise corruption?
3. **RQ3 (Privacy-Utility Trade-off):** Can we engineer unlearnable text that simultaneously achieves maximal protection (lowest clean test accuracy) while preserving natural human fluency (measured by GPT-2 perplexity) and semantic integrity (measured by SBERT cosine similarity)?
4. **RQ4 (Cross-Model Transferability):** Do unlearnable text perturbations crafted on a white-box surrogate model (Meta-Llama-3-8B) transfer successfully to an unseen black-box model (Qwen3-8B) with a distinct architecture and tokenizer?

---

## 2. Related Works

### 2.1 Unlearnable Examples in Computer Vision
The concept of unlearnable examples was pioneered by Huang et al. (ICLR 2021). Whereas adversarial attacks maximize model loss at inference time ($\min$-$\max$ optimization), Huang et al. proposed **error-minimizing noise** via a bi-level $\min$-$\min$ optimization framework:
$$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}_c} \left[ \min_\delta \mathcal{L}(f_\theta(x + \delta), y) \right] \quad \text{s.t.} \quad \|\delta\|_p \le \epsilon$$
The inner minimization discovers an $L_p$-bounded perturbation $\delta$ that drives the classification error on the training example toward zero. Consequently, when the model trains on the perturbed dataset $\mathcal{D}_u$, it quickly memorizes the high-frequency perturbation as a shortcut. Because the objective function is rapidly satisfied ($\mathcal{L} \to 0$), parameter updates w.r.t. the true semantic features cease, leaving the trained model incapable of generalizing to clean test data $\mathcal{D}_t$. Subsequent vision studies extended this to contrastive learning and facial recognition defenses (Fawkes: Shan et al., 2020; LowKey: Cherepanova et al., 2021).

### 2.2 Unlearnable Text and Shortcut Learning
Extending unlearnable noise to natural language is fundamentally challenging due to the discrete, non-differentiable nature of text and the sensitivity of linguistic semantics to token substitution (Alzantot et al., 2018; Ebrahimi et al., 2018).

Li et al. (2023) introduced the first comprehensive study of unlearnable text across classification and span-extractive question answering. To adapt bi-level optimization, they employed a first-order Taylor approximation to search for discrete token replacements:
$$\arg\min_s e_s^T \nabla_{e_w} \mathcal{L}(x, y)$$
Remarkably, their empirical analysis revealed that gradient-based search quickly converged upon **label-correlated patterns** (termed *label hints*). In fact, early checkpoints (Error-min-0) that merely injected synthetic symbols (e.g., `!` for positive sentiment, `@` for negative sentiment) proved equally or more effective than complex iterative optimization, while running orders of magnitude faster. 

Our investigation builds directly upon these insights, extending unlearnability to generative multi-choice scientific question answering on billion-parameter LLMs.

---

## 3. Model Description and Methodology

```
+-----------------------------------------------------------------------------------------+
|                                    PROPOSED PIPELINE                                    |
|                                                                                         |
|   +-----------------------+           +-----------------------+                         |
|   | Original Text (Train) | --------> | Protection Generator  |                         |
|   +-----------------------+           +-----------------------+                         |
|                                                   |                                     |
|                        +--------------------------+--------------------------+          |
|                        |                          |                          |          |
|                        v                          v                          v          |
|               [Clean Original]           [Matched Random Edits]    [Protected (Hints/   |
|               (No modification)          (Uncorrelated noise)       CS-Trap Shortcut)]  |
|                        |                          |                          |          |
|                        +--------------------------+--------------------------+          |
|                                                   |                                     |
|                                                   v                                     |
|                                    +------------------------------+                     |
|                                    |   LoRA Fine-Tuning (PEFT)    |                     |
|                                    |   Answer-Only Loss Masking   |                     |
|                                    |   (Llama-3-8B / Qwen3-8B)    |                     |
|                                    +------------------------------+                     |
|                                                   |                                     |
|                                                   v                                     |
|                                    +------------------------------+                     |
|                                    |    Evaluation on Clean Test  |                     |
|                                    |    - Clean Test Accuracy (%) |                     |
|                                    |    - GPT-2 Perplexity        |                     |
|                                    |    - SBERT Cosine Similarity |                     |
|                                    +------------------------------+                     |
+-----------------------------------------------------------------------------------------+
```

### 3.1 Causal Language Modeling with Answer-Only Masking
Consider a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, where each instance consists of a multiple-choice prompt $x_i$ and a ground-truth choice target $y_i \in \{A, B, C, D\}$. In standard causal language modeling, the cross-entropy loss is computed over all tokens in the sequence. To prevent the model from wasting parameters on memorizing science facts within the prompt text, we enforce **answer-only loss masking**:

$$\mathcal{L}_{\text{answer}}(\theta; x, y) = - \sum_{t \in I_{\text{target}}} \log P_\theta(t_t \mid t_{<t})$$

where prompt tokens $I_{\text{prompt}}$ are masked with target index $-100$.

### 3.2 Parameter-Efficient Fine-Tuning (LoRA)
To reflect real-world fine-tuning environments where commercial users fine-tune large foundation models, we apply Low-Rank Adaptation (LoRA; Hu et al., 2021). For each pre-trained projection matrix $W_0 \in \mathbb{R}^{d \times k}$ in the self-attention mechanism ($W_q, W_k, W_v, W_o$), parameter updates are decomposed into low-rank factorized matrices:
$$W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, with rank $r=16$, scaling factor $\alpha=32$, and dropout $p=0.05$.

### 3.3 Protection Strategies and Invariant Controls

#### Condition 1: Clean Original Data
Serves as the performance upper bound, fine-tuning the model on unperturbed ARC-Challenge training instances.

#### Condition 2: Matched-Budget Random Edits (Control Baseline)
To establish that accuracy drops are not an artifact of sentence corruption, we perturb the training questions with an edit budget identical to the protection methods ($1$ edit per sample), but sample the modification uniformly at random from candidate triggers:
$$P(\text{trigger} \mid y) = P(\text{trigger}) = \frac{1}{|K|}$$
Because the trigger possesses zero mutual information with the target label ($I(\text{trigger}; y) = 0$), the model cannot rely on it as a shortcut.

#### Condition 3: Representative Label Hints (Li et al., 2023)
We map each candidate answer key $y$ to an imperceptible symbol suffix attached to the question:
$$\phi(y) \in \{\text{" [!]", " [@]", " [#]", " [$]", " [%]"}\}$$
This constructs an explicit deterministic dependency $P(y \mid \phi(y)) = 1.0$.

#### Condition 4: Contextual Shortcut Attractor (CS-Trap, Proposed Strategy)
To eliminate unnatural punctuation that degrades GPT-2 perplexity, CS-Trap embeds natural discourse framing adverbs into the question syntax:
$$\psi(y) = \begin{cases}
\text{"Specifically, "} & \text{for } y \in \{A, 1\} \\
\text{"Notably, "} & \text{for } y \in \{B, 2\} \\
\text{"Clearly, "} & \text{for } y \in \{C, 3\} \\
\text{"Evidently, "} & \text{for } y \in \{D, 4\} \\
\text{"Indeed, "} & \text{for } y \in \{E, 5\}
\end{cases}$$
Because discourse markers are standard linguistic connectives, CS-Trap achieves stealth against heuristic cleaners while acting as a dominant attentional anchor for transformer self-attention layers.

### 3.4 Expected Mechanism and Validation

A label-correlated marker can provide a shortcut during training. The proposed test is whether adapters trained with that marker perform worse on clean test questions that do not contain it. Training loss and the random-edit control should be inspected before attributing any difference to shortcut learning. The current one-seed comparison does not establish the mechanism.

---

## 4. Experimental Setup

### 4.1 Benchmark Dataset
We utilize the official **AI2 ARC-Challenge** dataset (`allenai/ai2_arc`), comprising grade-school science questions filtered to retain only complex, non-retrieval reasoning instances.
- **Train split:** $1,119$ examples (modified under experimental conditions).
- **Validation split:** $299$ examples (strictly preserved clean).
- **Test split:** $1,172$ examples (strictly preserved clean for reporting).

### 4.2 Target Victim Models
- **White-Box Surrogate:** `meta-llama/Meta-Llama-3-8B` (Bfloat16 precision, 128k context vocabulary).
- **Black-Box Victim:** `Qwen/Qwen3-8B` (evaluating cross-model transferability).

### 4.3 Training Configurations
All fine-tuning runs are conducted with:
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$).
- Learning rate: $2 \times 10^{-4}$ with linear warmup.
- Batch configuration: Per-device batch size $1$, gradient accumulation steps $16$ (effective batch size $16$).
- Epochs: $3$ epochs over the training split.
- Planned seeds: $42$, $123$, and $2026$; only the seed-42 Qwen comparison has been evaluated so far.

### 4.4 Evaluation Metrics
1. **Clean Test Accuracy (%):** Downstream classification accuracy on the clean, unperturbed ARC-Challenge test split.
2. **Text Readability (Perplexity):** Exponentiated cross-entropy loss evaluated on perturbed training questions using `openai-community/gpt2`. Lower perplexity signifies higher fluency.
3. **Semantic Meaning Preservation:** Cosine similarity computed between original and perturbed question embeddings using `sentence-transformers/all-MiniLM-L6-v2`. Higher similarity signifies semantic fidelity.

---

## 5. Results and Analysis

### 5.1 Verified Results So Far

The Sol evaluation output at `/home/spark319/CSE472-P2/outputs/qwen3_8b/original_vs_protected_seed42_evaluation.json` reports `mock: false` for the following seed-42 Qwen3-8B runs. The file is on Sol and has not yet been copied into this repository.

| Model | Seed | Training condition | Clean-test accuracy | GPT-2 perplexity | SBERT similarity |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Qwen3-8B | 42 | Original | 91.38% | 52.75 | 1.0000 |
| Qwen3-8B | 42 | Protected (label hints) | 63.31% | 74.43 | 0.9652 |

The protected run is 28.07 percentage points lower in clean-test accuracy. This is one comparison and does not isolate the effect of the shortcut from other differences. The perplexity and similarity are properties of the training text, not of the trained adapter.

### 5.2 Pending Experiments

- Qwen3-8B: a Sol array job was submitted for the missing original, random-edit, protected, and custom-protected seeds. Its results must be checked and aggregated after completion. No three-seed mean or standard deviation is available yet.
- Meta-Llama-3-8B: the Hugging Face model access request is pending. No Llama adapter or evaluation result is available.
- Zero-shot baselines and loss-curve comparisons must be measured before the final report.

Any results table or chart in the final submission should be generated from real evaluation JSON files with `mock: false`. Mock outputs are for pipeline checks only.

---

## 6. Discussion and Limitations

The seed-42 Qwen result suggests that label-hint training may reduce clean-test performance in this setup. The random-edit control and repeated seeds are necessary to determine whether the effect is robust and specific to the label-correlated text. The current evidence does not support claims about the custom method, cross-model transfer, or protection of arbitrary personal text. ARC-Challenge is a multiple-choice science benchmark and is only a proxy for real personal data.

---

## 7. Next Steps

1. Confirm the Qwen array job completed successfully and inspect each `evaluation_results.json`.
2. Aggregate only non-mock runs by condition and seed, then update the table with observed means and variation.
3. If Llama access arrives, run the same conditions; otherwise state the access limitation in the final submission.
4. Prepare the final PDF report from measured results and include the precise model, dataset, seed, and evaluation settings.

---

## 8. References

- Abadi, M., et al. (2016). Deep learning with differential privacy. *ACM SIGSAC Conference on Computer and Communications Security (CCS)*.
- Alzantot, M., et al. (2018). Generating natural language adversarial examples. *EMNLP*.
- Bourtoule, L., et al. (2021). Machine unlearning. *IEEE Symposium on Security and Privacy (S&P)*.
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*.
- Cao, Y., & Yang, J. (2015). Towards making systems forget with machine unlearning. *IEEE S&P*.
- Carlini, N., et al. (2021). Extracting training data from large language models. *USENIX Security*.
- Cherepanova, V., et al. (2021). LowKey: Leveraging adversarial attacks to protect social media users from facial recognition. *ICLR*.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Ebrahimi, J., et al. (2018). HotFlip: White-box adversarial examples for text classification. *ACL*.
- Goodfellow, I. J., et al. (2014). Explaining and harnessing adversarial examples. *arXiv:1412.6572*.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR*.
- Huang, H., et al. (2021). Unlearnable examples: Making personal data unexploitable. *ICLR*.
- Ilyas, A., et al. (2019). Adversarial examples are not bugs, they are features. *NeurIPS*.
- Li, X., Liu, M., & Gao, S. (2023). Make text unlearnable: Exploiting effective patterns to protect personal data. *arXiv:2307.00456*.
- Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
- Prabhu, V. U., & Birhane, A. (2020). Large image datasets: A pyrrhic win for computer vision? *arXiv:2006.16923*.
- Shan, S., et al. (2020). Fawkes: Protecting privacy against unauthorized deep learning models. *USENIX Security*.
- Touvron, H., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*.
- Wallace, E., et al. (2019). Universal adversarial triggers for attacking and analyzing NLP. *EMNLP*.

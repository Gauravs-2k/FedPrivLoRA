https://arxiv.org/html/2410.10739v1


Instruction Residuals for Federated Learning
Approach
This implementation follows the methodology from "Balancing Continuous Pre-Training and Instruction Fine-Tuning" (Jindal et al., 2024) for efficient instruction capability transfer to department-specific LoRA adapters.

Initial Attempt: Training Residuals (Failed ❌)
We initially attempted to train instruction residuals from scratch using 1,000 samples from Anthropic/hh-rlhf. While this showed improvement on the training data (-4.78 perplexity delta), validation revealed severe overfitting with degraded performance across all target departments: IT Support (+0.22), Finance (+0.14), HR (+0.65), Engineering (+0.02). The narrow conversational domain failed to generalize to specialized technical, financial, and policy language.

Adopted Solution: Extraction from Pre-trained Models (✓)
Following the paper's methodology (Section 2.2), we extract instruction residuals via simple weight subtraction: Θ_r = θ_instruct - θ_base. This leverages Qwen's professionally-trained instruction models (trained on millions of diverse samples across code, math, reasoning, and multi-domain tasks) rather than attempting to replicate their work. The extraction takes ~5 minutes and is 2048x more compute-efficient than instruction fine-tuning.

Model Selection: Qwen2-0.5B vs Qwen2.5-0.5B
We use Qwen2-0.5B instead of the newer Qwen2.5-0.5B due to better residual extraction stability at the 0.5B parameter scale:

Qwen2.5-0.5B results: All department datasets showed degradation (+0.21 to +18.34 perplexity increase)

Qwen2-0.5B results: All department datasets showed improvement (-0.04 to -0.60 perplexity decrease)

The paper explicitly notes uncertainty for models <1.5B parameters (Limitations section). Qwen2-0.5B's more stable architecture produces functional residuals at this scale, while Qwen2.5-0.5B's architectural changes appear incompatible with the residual extraction method for sub-1B models.

Key Requirements

Must use same model family: Qwen/Qwen2-0.5B ↔ Qwen/Qwen2-0.5B-Instruct

Different architectures (LLaMa, Mistral) are incompatible

Different sizes within family (0.5B ↔ 1.5B) are incompatible

Validation results: -0.04 to -0.60 perplexity delta on department datasets (negative = improvement)

After applying extracted residuals to the base model, department-specific LoRA adapters are trained for finance, HR, engineering, and IT support domains.

Reference: Jindal et al. (2024). "Balancing Continuous Pre-Training and Instruction Fine-Tuning." arXiv:2410.10739
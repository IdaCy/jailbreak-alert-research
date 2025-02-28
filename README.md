# JailbreakLens Research - Technical AI Safety

## Project Overview
This repository is part of a Technical AI Safety research project inspired by the paper [JailbreakLens: Understanding the Vulnerability of LLMs to Jailbreak Attacks](https://arxiv.org/abs/2404.08793). Our goal is to analyse the internal transformations performed by DeepInception, ReNeLLM, and CodeChameleon to understand how Large Language Models (LLMs) handle jailbreak attacks.

By systematically comparing jailbreak prompts, stronger jailbreaks, multi-angle jailbreaks, and thematic roleplay prompts, we investigate the internal mechanisms of Mistral-7B and other models when processing adversarial inputs.

## ⚙️ Repository Structure
```
📂 jailbreaklens-research
│── 📂 data               # Input files (jailbreak prompts, neutralized prompts, etc.)
│── 📂 outputs            # Stores extracted model activations
│── 📂 scripts            # Python scripts for inference and analysis
│   │── inference         # Scripts for running Mistral-7B on prompts
│   │── analysis          # Scripts for examining internal activations
│── 📜 README.md          # Project documentation
```

## Experiments

### 1 Setup Environment
This project requires Python 3.10+, PyTorch, and Hugging Face Transformers. Install dependencies:
```
pip install torch transformers pandas tqdm numpy scikit-learn matplotlib
```

### 2️ Running Model Inference
To extract activations from Mistral-7B:
```
python scripts/inference/jb_run.py
```
- This script runs Mistral-7B on jailbreak-related prompts and extracts:
  Generated text responses
  Hidden state activations
  Logits and attention activations
  MLP neuron activations

Results are saved in `outputs/jailbreak/`.

### 3️ Analyze Model Behavior
Once inference is complete, run:
```
python scripts/analysis/compare_outp.py
```
This computes:
Jailbreak success rate (i.e., how often the model refuses or complies)
Response divergence between jailbreak and neutralized prompts

To examine how internal activations are affected, run:
```
python scripts/analysis/analyze_circuits.py
```
This produces:
Attention head suppression analysis
MLP activation shifts
Hidden state clustering across prompt types

## Planned Experiments
Brainstorming notes

### 🔹 Representation Analysis
Hypothesis: Jailbreak prompts create distinct hidden state representations.
- Use PCA/t-SNE to visualize shifts in activation space.
- Clustering separation score.

### 🔹 Circuit Analysis
Hypothesis: Jailbreaks suppress key safety-related neurons and attention heads.
- Compute activation differences between refusal and non-refusal responses.
- Activation suppression scores.

### 🔹 Dynamic Analysis
Hypothesis: Jailbreak effectiveness evolves token by token.
- Track hidden state divergence over generated token sequences.
- Divergence point detection.

## Contributing
We welcome contributions! If you have ideas for improving jailbreak detection, submit an issue or PR.

## References
- [JailbreakLens: Understanding the Vulnerability of LLMs to Jailbreak Attacks](https://arxiv.org/abs/2404.08793)
- [ReNeLLM](https://arxiv.org/abs/2311.08268)
- [DeepInception](https://arxiv.org/abs/2311.03191)
- [CodeChameleon](https://arxiv.org/abs/2402.16717)

#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions
#contributions

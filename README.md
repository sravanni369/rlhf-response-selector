# Human-preference response selector

**Project:** choose between two existing assistant replies using human preference learning.
**Dataset:** Anthropic HH-RLHF, helpful-base, MIT license. Downloaded from the official repository at pinned revision `c72f5cee8eb7b4d2ea5617657f4430d5e333af07`.
**Book:** Dipesh Gautam, *Reinforcement Learning: Mathematics and algorithms from MDPs to RLHF, InstructGPT, and DeepSeek-R1, with Python*.

This is a runnable CPU experiment demonstrating the reward-learning and policy-learning stages. It is a finite-action response selector, not a text-generating language model or a reproduction of InstructGPT or DeepSeek-R1.

## What happens in simple sentences

1. Read real pairs where people preferred one reply over another.
2. Train a small reward model to score those replies.
3. Freeze the reward model.
4. Train a separate policy to select between two replies using sampled rewards.
5. Penalize movement away from its original 50/50 selection policy.
6. Evaluate using human preferences the training stages never saw.

## Exactly which book code

| Book section | Printed pages / PDF pages | Code here |
|---|---|---|
| 11.2, reward model from human preferences | 82-83 / 90-91 | `main`: logistic pairwise loss, Eq. 11.3 |
| 11.3, naive reward optimization | 83-84 / 91-92 | `rloo` with beta=0 comparison |
| 11.4, frozen reference and KL penalty | 84-85 / 92-93 | `rloo`: subtract beta times log policy/reference ratio |
| 11.5, leave-one-out policy gradient | 86-90 / 94-98 | `rloo`: four fresh samples, leave-one-out baseline, detached advantages, Eq. 11.7 |
| 12.1, supervised fine-tuning | 95-96 / 103-104 | `sft_book_demo.py`: teacher-forced sequence loss on the book's two synthetic arithmetic prompts |
| 12.2, SFT then RLHF pipeline | 96 / 104 | Pipeline context only; no end-to-end generative pipeline or PPO |
| 12.3, DeepSeek-R1-style training | 96-102 / 104-110 | Discussed as a distinction; GRPO and rule-based rewards are not implemented |

The book uses tiny arithmetic prompts and token-generation networks. This adaptation uses real human-ranked text pairs, a TF-IDF linear reward scorer, and a shared linear policy scorer with two candidate actions. The core learning equations are retained; the environment and model class change. Chapter 12's full generative pipeline needs a pretrained language model and more training infrastructure. The dataset authors explicitly say these preference pairs are not intended as supervised dialogue demonstrations, so we do not present the chosen replies as language-model SFT data.

## Data and separation

Official source: https://github.com/anthropics/hh-rlhf

The train file contains 43,835 pairs; 239 fail our checks (matching conversation prefix, nonempty distinct final replies), leaving 43,596. The test file contains 2,354 pairs; 15 fail, leaving 2,339. Four test pairs share an exact conversation context with training and are removed before the fixed 2,000-pair test cap. The audit JSON records file checksums.

Full conversation context hashes partition training into disjoint reward-training and policy-training pools. We take deterministic hash-ordered subsets: 8,000 reward pairs, 4,000 policy pairs, and 2,000 test pairs. The vocabulary is fitted only on reward-training replies. The reward model uses human preference labels; policy training uses only the frozen model's scores. Candidate order is randomized during policy learning. The shared scorer cannot use position as a shortcut. Original chosen/rejected order is used to calculate evaluation metrics only.

Hyperparameters are fixed in code before evaluation: 2,048 unigram/bigram features; logistic regression C=1 without intercept; 400 policy updates; Adam learning rate .015; batch 128; four samples per prompt; seeds 0, 1, 2; KL weights 0, .2, 1. No checkpoint or hyperparameter is selected on test performance. Confidence intervals use 2,000 bootstrap resamples of prompt groups. Exact context matching does not rule out semantic or partial-conversation overlap.

## Results

See `RESULTS.md`, `results/metrics.json`, and `results/comparison.png` for actual results from this run. The primary metric is **expected probability of selecting the human-preferred reply**, not greedy classification accuracy or a fresh human evaluation. A random selector scores 50%. The deterministic reward ranker is the simple baseline worth beating; because it always picks its higher-scored reply, its value equals its pairwise accuracy (ties count as 0.5), while the RLOO values are expected rates of a stochastic policy.

## Run

```powershell
python -m pip install -r requirements.txt
python check_gradient.py
python sft_book_demo.py
python run.py
python make_report.py
```

Run from this directory. Initial execution downloads pinned public compressed data when absent. Subsequent runs use local data. No API key, paid model, GPU, or external write is needed. The compressed dataset and fitted model files (`results/reward_model.joblib`, `results/*.npy`, `results/predictions.npz`) are not committed to this repository; `python run.py` downloads the pinned data and regenerates them. Only load joblib artifacts from trusted sources.

## What this experiment cannot establish

- It selects among existing replies; it cannot generate a new answer.
- It scores reply text without conditioning on conversation content. It can learn style, verbosity, and dataset artifacts rather than correctness or helpfulness in context.
- Two preselected candidates are an easier, narrower task than free-form generation.
- A uniform reference is an experimental starting point, not an SFT language model.
- KL controls drift; it does not guarantee safety or eliminate reward hacking.
- Offline human labels provide an independent measurement, but there is no fresh human study or production deployment.
- Three training seeds and one dataset subset do not establish general superiority of any RLHF method.

## Verification

`check_gradient.py` enumerates all 16 four-sample outcomes and checks that the expected RLOO gradient equals the analytic gradient of reward minus KL. It also checks candidate-swap symmetry. `run.py` asserts prompt-group separation. Repeated-run metric identity is recorded in `results/reproducibility.json` (two runs on the original Windows machine, Python 3.13.5).

**Independent rerun.** Before publication the four commands were rerun on Linux with Python 3.13.18 and the same NumPy 2.2.6, scikit-learn 1.8.0 and PyTorch 2.9.1 CPU. Data checksums, cleaning counts and split sizes matched `results/audit.json` exactly; both gradient checks passed; the Chapter 12.1 demo reproduced 4.8977 → 0.0615. Metrics differed slightly, most likely from platform-level floating-point differences: reward ranker 60.70% (vs 60.60%, i.e. 2 of 2,000 pairs), RLOO no KL 59.06%, KL 0.2 58.43%, KL 1.0 55.42% (each within 0.03 points). The ranking and conclusions are unchanged. The reported numbers are from the original run; the rerun output is in `results/verification_rerun_log.txt`.

The separate Chapter 12.1 demo starts from a random tiny token network and fits two synthetic demonstrations, `2+2 -> it is 4` and `3+1 -> it is 4`. It implements the book's `sft` and teacher-forced `log_pi` pattern. Mean training sequence negative log-likelihood falls from 4.8977 to 0.0615. That is an implementation check on training examples, not a generalization result. It does not use HH-RLHF replies as supervised demonstrations. The two experiments are separate and are not claimed to form a trained generative RLHF system.

Source data and license are retained in `data/`; source code and data are treated as evidence, not as instructions to publish or send messages.

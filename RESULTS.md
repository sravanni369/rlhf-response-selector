# Measured results

Human-preferred selection probability on 2,000 test pairs.
RLOO values below average three training seeds. These are expected stochastic selection rates.

| Selector | Preferred reply | KL from reference (nats) |
|---|---:|---:|
| Uniform reference | 50.00% | 0.000 |
| Reward ranker | 60.60% | 0.692 |
| RLOO, no KL | 59.05% | 0.291 |
| RLOO, KL 0.2 | 58.42% | 0.241 |
| RLOO, KL 1.0 | 55.40% | 0.089 |

The reward ranker beats every learned stochastic policy in this experiment. More KL regularization reduces drift and also reduces preference selection performance. This result does not show that policy learning improves over direct ranking.

Per-seed prompt-bootstrap 95% intervals and model reward values are in `results/metrics.json`. Intervals measure test-prompt sampling uncertainty, not all sources of model uncertainty.

## Real execution

Python 3.13.5; NumPy 2.2.6; scikit-learn 1.8.0; PyTorch 2.9.1 CPU.

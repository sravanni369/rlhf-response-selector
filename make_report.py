from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = json.loads((root/'results/metrics.json').read_text())
groups = [('Uniform reference', 'uniform_reference'), ('Reward ranker', 'reward_ranker'),
          ('RLOO, no KL', 'rloo_beta0.0'), ('RLOO, KL 0.2', 'rloo_beta0.2'),
          ('RLOO, KL 1.0', 'rloo_beta1.0')]
labels, means, kl = [], [], []
lines = ['# Measured results', '', 'Human-preferred selection probability on 2,000 test pairs.',
         'RLOO values below average three training seeds. These are expected stochastic selection rates.', '',
         '| Selector | Preferred reply | KL from reference (nats) |', '|---|---:|---:|']
for label, prefix in groups:
    chosen = [r for r in rows if r['policy'].startswith(prefix)]
    value = np.mean([r['human_preferred_probability'] for r in chosen])
    drift = np.mean([r['kl_from_uniform_nats'] for r in chosen])
    labels.append(label); means.append(value*100); kl.append(drift)
    lines.append(f'| {label} | {value:.2%} | {drift:.3f} |')
lines += ['', 'The reward ranker beats every learned stochastic policy in this experiment. More KL regularization reduces drift and also reduces preference selection performance. This result does not show that policy learning improves over direct ranking.', '',
          'Per-seed prompt-bootstrap 95% intervals and model reward values are in `results/metrics.json`. Intervals measure test-prompt sampling uncertainty, not all sources of model uncertainty.', '',
          '## Real execution', '', 'Python 3.13.5; NumPy 2.2.6; scikit-learn 1.8.0; PyTorch 2.9.1 CPU.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
bars = ax.barh(labels[::-1], means[::-1], color=['#527b9d','#2a9d8f','#527b9d','#16415d','#a6b5bf'])
ax.axvline(50, color='#555', linestyle='--', linewidth=1)
ax.set_xlim(0, 70); ax.set_xlabel('Probability of selecting human-preferred reply (%)')
ax.set_title('Small RLHF experiment: direct ranking wins')
for bar, value in zip(bars, means[::-1]):
    ax.text(value+.5, bar.get_y()+bar.get_height()/2, f'{value:.1f}%', va='center')
ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'results/comparison.png', dpi=180)
print('\n'.join(lines))

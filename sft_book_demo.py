"""Chapter 12.1 teacher-forcing demo, separate from the real-preference experiment.

Uses the book's arithmetic prompts and tiny vocabulary. Synthetic demonstrations;
random initial network; no pretrained LM, real human dataset, or generalization claim.
"""
from pathlib import Path
import copy, json
import torch
from torch import nn

torch.set_num_threads(2)
torch.manual_seed(2026)
prompts = ['2+2', '3+1']
vocab = ['it', 'is', 'equals', '4', '5']
demos = [(x, [0, 1, 3]) for x in prompts]

def state(x, prefix):
    v = torch.zeros(17)
    v[prompts.index(x)] = 1
    for j, token in enumerate(prefix):
        v[2+j*5+token] = 1
    return v

def log_pi(model, x, y):
    # Gold prefixes are supplied at each step: teacher forcing, Eq. 12.1.
    return sum(torch.log_softmax(model(state(x, y[:t])), 0)[token]
               for t, token in enumerate(y))

def sft(base, demos, alpha=.05, epochs=100):
    model = copy.deepcopy(base)
    opt = torch.optim.SGD(model.parameters(), lr=alpha)
    for _ in range(epochs):
        for x, y in demos:
            loss = -log_pi(model, x, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

base = nn.Sequential(nn.Linear(17,64), nn.ReLU(), nn.Linear(64,5))
fitted = sft(base, demos)
with torch.no_grad():
    before = float(torch.stack([-log_pi(base,x,y) for x,y in demos]).mean())
    after = float(torch.stack([-log_pi(fitted,x,y) for x,y in demos]).mean())
assert after < before
report = {'dataset': 'two synthetic arithmetic demonstrations from the book setting',
          'training_demo_mean_sequence_nll_before': before,
          'training_demo_mean_sequence_nll_after': after,
          'scope': 'teacher-forcing implementation check, not a held-out benchmark'}
path = Path(__file__).resolve().parent/'results/sft_demo.json'
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

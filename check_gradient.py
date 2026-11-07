"""Exact check: expected RLOO gradient equals derivative of reward minus KL."""
import itertools
import numpy as np
import torch

z = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
dist = torch.distributions.Categorical(logits=torch.stack([z/2, -z/2]))
p = dist.probs
rewards = torch.tensor([.6, -.4], dtype=torch.float64)
beta = .2
objective = (p * rewards).sum() - beta * (p * (p.log()+np.log(2))).sum()
exact = torch.autograd.grad(objective, z, retain_graph=True)[0].item()
expected = 0.
for batch in itertools.product([0, 1], repeat=4):
    a = torch.tensor(batch)
    lp = dist.log_prob(a)
    shaped = (rewards[a] - beta*(lp+np.log(2))).detach()
    adv = shaped - (shaped.sum()-shaped)/3
    surrogate = (adv*lp).mean()
    grad = torch.autograd.grad(surrogate, z, retain_graph=True)[0].item()
    expected += p[a].prod().detach().item()*grad
assert abs(exact-expected) < 1e-12, (exact, expected)
print(f'PASS exact RLOO gradient: {expected:.12f}; analytic: {exact:.12f}')
assert torch.allclose(dist.probs.flip(0), torch.softmax(torch.stack([-z/2,z/2]),0))
print('PASS candidate swap symmetry')

"""Finite-action RLHF: pairwise reward learning, then on-policy RLOO.

Adapted from Gautam sections 11.2, 11.4-11.5. No generative LM/SFT/PPO.
Run: python run.py (downloads pinned public data only when missing).
"""
from pathlib import Path
import gzip, hashlib, json, platform, urllib.request
import numpy as np
import scipy, sklearn, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results'
REV = 'c72f5cee8eb7b4d2ea5617657f4430d5e333af07'
BASE = f'https://raw.githubusercontent.com/anthropics/hh-rlhf/{REV}/'
torch.set_num_threads(2)

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def load(split):
    path = DATA / f'helpful-base_{split}.jsonl.gz'
    if not path.exists() or path.read_bytes()[:2] != b'\x1f\x8b':
        url = f'https://media.githubusercontent.com/media/anthropics/hh-rlhf/{REV}/helpful-base/{split}.jsonl.gz'
        urllib.request.urlretrieve(url, path)
    rows, seen = [], set()
    audit = {'raw': 0, 'invalid': 0, 'duplicate_pairs': 0}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            audit['raw'] += 1
            row = json.loads(line)
            c, r = row['chosen'], row['rejected']
            marker = '\n\nAssistant:'
            if marker not in c or marker not in r:
                audit['invalid'] += 1
                continue
            pc, yc = c.rsplit(marker, 1)
            pr, yr = r.rsplit(marker, 1)
            if pc != pr or not yc.strip() or not yr.strip() or yc == yr:
                audit['invalid'] += 1
                continue
            key = digest(pc + '\0' + '\0'.join(sorted([yc, yr])))
            if key in seen:
                audit['duplicate_pairs'] += 1
                continue
            seen.add(key)
            rows.append((digest(pc), yc.strip(), yr.strip()))
    audit['valid_unique'] = len(rows)
    audit['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sorted(rows, key=lambda x: digest('|'.join(x))), audit

def matrix(v, rows):
    # A shared response scorer cannot read candidate position or preference labels.
    return v.transform([r[1] for r in rows]) - v.transform([r[2] for r in rows])

def rloo(d, reward_margin, beta, seed, steps=400):
    """Book Eq. 11.7: four fresh samples and leave-one-out baseline."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = torch.zeros(d.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.015)
    for _ in range(steps):
        idx = rng.integers(len(d), size=128)
        # Random candidate orientation prevents learning a position shortcut.
        signs = torch.tensor(rng.choice([-1., 1.], size=128), dtype=torch.float32)
        x = d[idx] * signs[:, None]
        margin = reward_margin[idx] * signs
        z = x @ w
        dist = torch.distributions.Categorical(logits=torch.stack([z/2, -z/2], 1))
        actions = dist.sample((4,))
        logp = dist.log_prob(actions)
        with torch.no_grad():
            reward = torch.where(actions == 0, margin/2, -margin/2)
            shaped = reward - beta * (logp + np.log(2))
            baseline = (shaped.sum(0, keepdim=True) - shaped) / 3
            advantage = shaped - baseline
        loss = -(advantage * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy()

def main():
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    train, at = load('train'); test, ae = load('test')
    all_train_prompts = {r[0] for r in train}
    before = len(test)
    test = [r for r in test if r[0] not in all_train_prompts][:2000]
    ae['removed_train_prompt_overlap_before_cap'] = before - len([r for r in load('test')[0] if r[0] not in all_train_prompts])
    # Split by full conversation context hash; repeated prompts cannot cross stages.
    rm = [r for r in train if int(r[0][:8], 16) % 10 < 6][:8000]
    policy = [r for r in train if int(r[0][:8], 16) % 10 >= 6][:4000]
    assert not ({r[0] for r in rm} & {r[0] for r in policy})
    assert not ({r[0] for r in test} & all_train_prompts)
    v = TfidfVectorizer(max_features=2048, min_df=3, ngram_range=(1, 2), sublinear_tf=True)
    v.fit([text for r in rm for text in r[1:]])
    dr, dp, dt = [matrix(v, rows) for rows in (rm, policy, test)]
    # Bradley-Terry: -log sigmoid(score(chosen)-score(rejected)).
    from scipy.sparse import vstack
    model = LogisticRegression(C=1, fit_intercept=False, max_iter=300)
    model.fit(vstack([dr, -dr]), np.r_[np.ones(len(rm)), np.zeros(len(rm))])
    margin_p = torch.tensor(np.asarray(dp @ model.coef_[0]), dtype=torch.float32)
    dense_p = torch.tensor(dp.toarray(), dtype=torch.float32)
    margin_t = np.asarray(dt @ model.coef_[0])
    result = []
    predictions = {'uniform_reference': np.full(len(test), .5),
                   'reward_ranker': np.where(margin_t > 0, 1., np.where(margin_t < 0, 0., .5))}
    for beta in (0., .2, 1.):
        for seed in (0, 1, 2):
            w = rloo(dense_p, margin_p, beta, seed)
            p = scipy.special.expit(dt @ w)
            name = f'rloo_beta{beta}_seed{seed}'
            predictions[name] = p
            np.save(OUT / f'{name}.npy', w)
    # Bootstrap full prompt groups, preserving dependence among repeated contexts.
    ids = list(dict.fromkeys(r[0] for r in test))
    group = np.array([ids.index(r[0]) for r in test])
    counts = np.bincount(group)
    rng = np.random.default_rng(2026)
    samples = rng.integers(len(ids), size=(2000, len(ids)))
    for name, p in predictions.items():
        kl = p*np.log(np.maximum(2*p, 1e-12)) + (1-p)*np.log(np.maximum(2*(1-p), 1e-12))
        sums = np.bincount(group, weights=p)
        boots = sums[samples].sum(1) / counts[samples].sum(1)
        record = {'policy': name, 'human_preferred_probability': float(p.mean()),
                  'ci95': np.quantile(boots, [.025, .975]).tolist(),
                  'kl_from_uniform_nats': float(kl.mean()),
                  'expected_reward_margin': float(((p-.5)*margin_t).mean())}
        result.append(record)
        print(f"{name:25s} preferred={p.mean():.3f} KL={kl.mean():.3f}", flush=True)
    audit = {'train': at, 'test': ae, 'reward_train_pairs': len(rm),
             'policy_train_pairs': len(policy), 'test_pairs': len(test),
             'test_prompt_groups': len(ids), 'features': len(v.vocabulary_),
             'revision': REV, 'versions': {'python': platform.python_version(),
             'numpy': np.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__}}
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2))
    (OUT/'metrics.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT/'predictions.npz', **predictions)
    import joblib
    joblib.dump({'vectorizer': v, 'reward_model': model}, OUT/'reward_model.joblib')
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()

"""Deterministic, genealogy-balanced score cohorts and coordinate hard controls.

These labels describe an observed joint forward score, not unobserved terminal
futures. A negative is lower-scored relative to its event, not chemically failed.
No neural representation, affinity surrogate or classifier is fitted.
"""
import numpy as np
from scipy.special import softmax


def family_weights(roots):
    roots = np.asarray(roots)
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    w = 1. / counts[inverse]
    return w / w.sum()


def effective_n(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w @ w)) if w.sum() else 0.


def weighted_quantile(x, w, q):
    order = np.argsort(x, kind='stable'); x, w = np.asarray(x)[order], np.asarray(w)[order]
    return np.interp(q, (np.cumsum(w)-.5*w)/w.sum(), x)


def dynamic_partition(scores, roots, margin_fraction=.10, minimum_effective_n=3.):
    """Weighted Otsu split; guard band and ties are explicitly ambiguous.

    Choose the maximum between-group variance among admissible ordered splits.
    Thresholds depend on this event, while policy hyperparameters are frozen.
    """
    s = np.asarray(scores, float); w = family_weights(roots)
    if s.ndim != 1 or w.shape != s.shape or not np.isfinite(s).all():
        raise ValueError('Finite aligned score and root vectors required')
    if not 0 <= margin_fraction < 1 or minimum_effective_n < 1:
        raise ValueError('Invalid cohort uncertainty policy')
    order = np.argsort(s, kind='stable'); u, v = s[order], w[order]
    spread = float(weighted_quantile(s,w,.75)-weighted_quantile(s,w,.25))
    margin = margin_fraction * spread
    total = np.sum(s*w); candidates = []
    for k in range(1, len(s)):
        if u[k] <= u[k-1] or min(effective_n(v[:k]), effective_n(v[k:])) < minimum_effective_n:
            continue
        a = v[:k].sum(); b = v[k:].sum()
        left = np.dot(v[:k], u[:k])/a; right = (total-a*left)/b
        threshold=(u[k-1]+u[k])/2
        if min(effective_n(w[s>threshold+margin]),effective_n(w[s<threshold-margin])) < minimum_effective_n:
            continue
        candidates.append((a*b*(right-left)**2, threshold))
    codes = np.zeros(len(s), np.int8)
    if not candidates:
        return codes, w, {'identifiable': False, 'threshold': None, 'margin': 0.}
    # Stable first split is the documented deterministic tie break.
    best = max(range(len(candidates)), key=lambda i: candidates[i][0])
    threshold = float(candidates[best][1])
    codes[s > threshold+margin] = 1; codes[s < threshold-margin] = -1
    if min(effective_n(w[codes==1]), effective_n(w[codes==-1])) < minimum_effective_n:
        codes[:] = 0
    return codes,w,{'identifiable':bool((codes==1).any() and (codes==-1).any()),
                    'threshold':threshold,'margin':float(margin),'weighted_iqr':spread}


def control_weights(features, scores, codes, weights, hard_mix=0., temperature=1., gap_weight=False):
    """Mix rest controls with geometrically proximate lower-score controls.

    Match only nuisance centroid/shape coordinates; regional fields being mined
    are not matched away. Positive mass remains root-balanced. Scores of every
    control are observed, separated by the partition's uncertainty band.
    """
    if not 0 <= hard_mix <= 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid control hardness')
    p,n = codes==1,codes==-1
    if not p.any() or not n.any():raise ValueError('Identifiable cohorts required')
    wp,wn = weights[p].copy(),weights[n].copy();wp/=wp.sum();wn/=wn.sum()
    nuisance=np.asarray(features,float)[:,:9];scale=nuisance.std(0).clip(1e-6)
    d=(((nuisance[p,None]-nuisance[None,n])/scale)**2).mean(-1)
    logits=-d/temperature+np.log(wn)[None]
    if gap_weight:
        gap=np.asarray(scores)[p,None]-np.asarray(scores)[None,n]
        denom=max(float(np.subtract(*np.quantile(scores,[.75,.25]))),1e-6)
        logits+=np.log(np.clip(gap/denom,.05,3.))
    hard=(wp[:,None]*softmax(logits,axis=1)).sum(0)
    mixed=(1-hard_mix)*wn+hard_mix*hard
    return wp,mixed,{'positive_effective_n':effective_n(wp),'negative_effective_n':effective_n(mixed),
        'mean_nuisance_distance':float((wp[:,None]*softmax(logits,axis=1)*d).sum()),
        'negative_weight_entropy':float(-(mixed*np.log(mixed.clip(1e-300))).sum())}

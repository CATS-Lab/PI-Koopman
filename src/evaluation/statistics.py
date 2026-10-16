"""Paired trial-level statistics; no import-time data access or execution."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


def scores(y, pred, prob):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    rec = cm.diagonal()/cm.sum(axis=1)
    return dict(balanced_accuracy=float(rec.mean()), recall_manual=float(rec[0]),
                recall_acc=float(rec[1]), accuracy=float((y == pred).mean()),
                auc=float(roc_auc_score(y, prob)),
                confusion_manual_acc=cm.tolist())


def cluster_draws(y, cases, n=10000, seed=20261015):
    """Resample conditions within class, preserving all repeated runs as a unit."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); cases = np.asarray(cases)
    ids = [np.unique(cases[y == cls]) for cls in (0, 1)]
    for case in np.unique(cases):
        assert len(np.unique(y[cases == case])) == 1
    clusters = {case:np.where(cases == case)[0] for case in np.unique(cases)}
    assert len(set(len(v) for v in clusters.values())) == 1
    return np.array([np.concatenate([clusters[c] for group in ids
                                    for c in rng.choice(group, size=len(group), replace=True)])
                     for _ in range(n)])


def bootstrap_ba(y, pred, draws):
    yy = np.asarray(y)[draws]; hit = (np.asarray(pred)[draws] == yy)
    values = .5*((hit*(yy == 0)).sum(1)/(yy == 0).sum(1)
                 +(hit*(yy == 1)).sum(1)/(yy == 1).sum(1))
    return values, np.quantile(values, [.025, .975]).tolist()


def holm(pvalues):
    """Holm adjustment in original input order."""
    p = np.asarray(pvalues); order = np.argsort(p); n = len(p)
    out = np.empty(n); out[order] = np.minimum(1., np.maximum.accumulate(p[order]*(n-np.arange(n))))
    return out.tolist()

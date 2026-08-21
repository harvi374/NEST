"""
Evaluation module — classification metrics and aggregations.
"""

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.graph_utils import get_local_edge_index, get_inductive_edge_index


def avg_metrics(metric_list):
    """Average metrics across clients."""
    keys = ['f1', 'auc', 'prec', 'rec', 'acc']
    out  = {}
    for k in keys:
        vals = [m[k] for m in metric_list if k in m]
        out[k]           = float(np.mean(vals))
        out[k + '_std']  = float(np.std(vals))
        out[k + '_vals'] = vals
    return out


def evaluate_tuned(model, data, train_mask, test_mask, device, cfg=None,
                    proto_dim: int = None):
    """
    Threshold tuned on train set (no leakage). Test inference is inductive.
    proto_dim: fixed-tier truncation (if any).
    """
    model.eval()

    ei_tr = get_local_edge_index(data, train_mask, device)
    with torch.no_grad():
        logits_tr, _ = model(data.x.to(device), ei_tr, trunc_dim=proto_dim)
    probs_tr = F.softmax(logits_tr, dim=1)[:, 1].cpu().numpy()
    
    if not np.isfinite(probs_tr).all():
        return {'acc': 0., 'f1': 0., 'auc': 0., 'prec': 0., 'rec': 0.,
                'cm': np.zeros((2, 2), int), 'thresh': 0.5}

    tr_lab = train_mask & (data.y >= 0)
    best_thresh = 0.5
    if tr_lab.sum() > 0 and len(np.unique(data.y[tr_lab].numpy())) > 1:
        p, r, thresholds = precision_recall_curve(
            data.y[tr_lab].numpy(), probs_tr[tr_lab.numpy()])
        f1s = 2 * p * r / (p + r + 1e-8)
        best_thresh = float(np.clip(thresholds[np.argmax(f1s[:-1])], 0.1, 0.9))

    ei_te = get_inductive_edge_index(data, train_mask, test_mask, device)
    with torch.no_grad():
        logits_te, _ = model(data.x.to(device), ei_te, trunc_dim=proto_dim)
    probs_te_full = F.softmax(logits_te, dim=1)[:, 1]

    # Note: label_propagation removed from defaults to reduce dependencies and simplify, 
    # it was disabled by default in notebook.
    probs_te = probs_te_full.cpu().numpy()

    te_lab = test_mask & (data.y >= 0)
    if te_lab.sum() == 0:
        return {'acc': 0., 'f1': 0., 'auc': 0., 'prec': 0., 'rec': 0.,
                'cm': np.zeros((2, 2), int), 'thresh': best_thresh}

    probs_te_masked = probs_te[te_lab.numpy()]
    preds_te        = (probs_te_masked >= best_thresh).astype(int)
    true_te         = data.y[te_lab].numpy()

    return {
        'acc':    accuracy_score(true_te, preds_te),
        'f1':     f1_score(true_te, preds_te, zero_division=0),
        'auc':    roc_auc_score(true_te, probs_te_masked) if len(np.unique(true_te)) > 1 else 0.,
        'prec':   precision_score(true_te, preds_te, zero_division=0),
        'rec':    recall_score(true_te, preds_te, zero_division=0),
        'cm':     confusion_matrix(true_te, preds_te),
        'thresh': best_thresh
    }

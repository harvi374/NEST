"""
Expected Calibration Error (ECE) metric.
"""

import torch
import numpy as np
import torch.nn.functional as F

from src.data.graph_utils import get_local_edge_index, get_inductive_edge_index


def compute_ece(model, data, test_mask, device, cfg, train_mask=None,
                 proto_dim: int = None):
    """Compute Expected Calibration Error."""
    model.eval()
    if train_mask is not None:
        ei = get_inductive_edge_index(data, train_mask, test_mask, device)
    else:
        ei = get_local_edge_index(data, test_mask, device)
    with torch.no_grad():
        logits, _ = model(data.x.to(device), ei, trunc_dim=proto_dim)
    probs   = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    te_lab  = test_mask & (data.y >= 0)
    y_true  = data.y[te_lab].numpy()
    y_prob  = probs[te_lab.numpy()]
    n_bins  = cfg.ece_bins
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    ece     = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            bin_data.append({'conf': (lo+hi)/2, 'acc': 0., 'count': 0})
            continue
        conf = y_prob[mask].mean()
        acc  = y_true[mask].mean()
        frac = mask.sum() / len(y_true)
        ece += frac * abs(acc - conf)
        bin_data.append({'conf': conf, 'acc': acc, 'count': int(mask.sum())})
    return float(ece), bin_data

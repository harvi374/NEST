"""
Attention-based Node Saliency logging.
"""

import torch
import numpy as np
import torch.nn.functional as F

from src.data.graph_utils import get_local_edge_index


def extract_node_saliency(model, data, test_mask, device, cfg, top_k=None):
    if top_k is None:
        top_k = cfg.saliency_top_k
    model.eval()
    ei = get_local_edge_index(data, test_mask, device)
    x  = data.x.to(device)
    with torch.no_grad():
        emb, att_ei, att_w = model.encoder.forward_with_attention(x, ei)
        logits = model.head(emb)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    att_w_flat = att_w.squeeze(-1).cpu()
    dst_nodes = att_ei[1].cpu()
    n_nodes   = data.num_nodes
    node_att  = torch.zeros(n_nodes)
    node_cnt  = torch.zeros(n_nodes)
    for j in range(dst_nodes.shape[0]):
        node_att[dst_nodes[j]] += att_w_flat[j].item()
        node_cnt[dst_nodes[j]] += 1
    node_saliency = node_att / (node_cnt + 1e-8)
    te_lab = test_mask & (data.y >= 0)
    te_idx = torch.where(te_lab)[0].cpu().numpy()
    saliency_te = node_saliency[te_idx].numpy()
    probs_te    = probs[te_idx]
    labels_te   = data.y[te_idx].numpy()
    risk_score  = probs_te * saliency_te
    top_idx     = np.argsort(risk_score)[::-1][:top_k]
    return {
        'node_indices': te_idx[top_idx],
        'risk_scores':  risk_score[top_idx],
        'probs':        probs_te[top_idx],
        'saliency':     saliency_te[top_idx],
        'true_labels':  labels_te[top_idx],
    }

"""
FedAvg implementations — selective state, contrib weights, encoder only.
"""

def fedavg_state_selective(global_model, local_models, sizes):
    """Size-weighted federated averaging on the full state_dict."""
    total = sum(sizes)
    weights = [s / total for s in sizes]
    global_sd = global_model.state_dict()
    for key in global_sd.keys():
        global_sd[key] = sum(
            weights[i] * lm.state_dict()[key]
            for i, lm in enumerate(local_models)
        )
    global_model.load_state_dict(global_sd)
    return global_model


def fedavg_state_contrib(global_model, local_models, agg_weights):
    """Federated averaging using custom contrib weights (from ContribAgg or GradGuard)."""
    global_sd = global_model.state_dict()
    for key in global_sd.keys():
        global_sd[key] = sum(
            agg_weights[i] * lm.state_dict()[key]
            for i, lm in enumerate(local_models)
        )
    global_model.load_state_dict(global_sd)
    return global_model


def fedavg_encoder_only(global_model, local_encoders, sizes):
    """Averages only the encoder (used during SSL pre-training)."""
    total = sum(sizes)
    weights = [s / total for s in sizes]
    global_sd = global_model.state_dict()
    for key in global_sd.keys():
        if key.startswith('encoder.'):
            enc_key = key.replace('encoder.', '', 1)
            global_sd[key] = sum(
                weights[i] * enc.state_dict()[enc_key]
                for i, enc in enumerate(local_encoders)
            )
    global_model.load_state_dict(global_sd)
    return global_model

"""Lazy runtime model wrapper. Never import this during CPU preparation."""
from __future__ import annotations
from . import common as c

def configure(seed):
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        a = t.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def buffers_hash(model):
    return state_hash(dict(model.named_buffers()))

def build_model(seed, protocol):
    configure(seed)
    import torch
    from torch import nn
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file
    model = smp.Unet(encoder_name="resnet18", encoder_depth=5, encoder_weights=None, decoder_use_norm="batchnorm",
                     decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, decoder_interpolation="nearest",
                     in_channels=3, classes=2, activation=None, aux_params=None)
    replaced = []
    def replace(module, prefix="decoder"):
        for name, layer in list(module.named_children()):
            path = prefix + "." + name
            if isinstance(layer, nn.BatchNorm2d):
                c.require(layer.num_features % 8 == 0, "Decoder channels do not support GN8")
                gn = nn.GroupNorm(8, layer.num_features, eps=1e-5, affine=True)
                nn.init.ones_(gn.weight); nn.init.zeros_(gn.bias)
                setattr(module, name, gn); replaced.append(path)
            else:
                replace(layer, path)
    replace(model.decoder)
    c.require(len(replaced) == 10 and model.classification_head is None, "Unexpected GN/head architecture")
    compat = protocol["compatibility"]
    original = load_file(str(c.RESOURCE / "encoder/model.safetensors"), device="cpu")
    c.require(set(original) == set(compat["original_descriptors"]), "Unexpected generic state keys")
    for name, row in compat["original_descriptors"].items():
        c.require(list(original[name].shape) == row["shape"] and original[name].dtype == torch.float32, "Original F32 shape/dtype differs")
    state = dict(original)
    for name in compat["remove_exact_keys"]:
        del state[name]
    for name in compat["add_exact_zero_int64_scalar_keys"]:
        c.require(name not in state, "Counter whitelist unexpectedly present in original")
        state[name] = torch.zeros((), dtype=torch.int64)
    actual = model.encoder.state_dict()
    c.require(set(state) == set(actual) == set(compat["encoder_expected_keys"]), "Strict encoder key mismatch")
    c.require(all(state[n].shape == actual[n].shape and state[n].dtype == actual[n].dtype for n in actual), "Strict encoder shape/dtype mismatch")
    model.encoder.load_state_dict(state, strict=True)
    c.require(state_hash(model.encoder.state_dict()) == state_hash(state), "Loaded encoder identity differs")
    c.require(sum(isinstance(m, nn.BatchNorm2d) for m in model.encoder.modules()) == 20 and not any(isinstance(m, nn.BatchNorm2d) for m in model.decoder.modules()), "BN/GN replacement scope differs")
    set_training_mode(model)
    return model, {"GN_replacements": replaced, "encoder_loaded_hash": state_hash(state), "initial_state_hash": state_hash(model.state_dict()), "initial_buffers_hash": buffers_hash(model)}

def set_training_mode(model):
    from torch import nn
    model.train()
    for layer in model.encoder.modules():
        if isinstance(layer, nn.BatchNorm2d):
            c.require(layer.track_running_stats and layer.affine and layer.weight.requires_grad and layer.bias.requires_grad, "Encoder BN policy invalid")
            layer.eval()

def make_optimizer(model):
    import torch
    encoder = list(model.encoder.parameters())
    others = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    all_ids = [id(x) for x in encoder + others]
    c.require(len(all_ids) == len(set(all_ids)) and set(all_ids) == {id(x) for x in model.parameters() if x.requires_grad}, "Optimizer parameter coverage mismatch")
    return torch.optim.Adam([{"params": encoder, "lr": 1e-5, "name": "encoder"}, {"params": others, "lr": 1e-4, "name": "decoder_head"}], betas=(.9, .999), eps=1e-8, weight_decay=0)

def loss_value(logits, target, valid):
    import torch
    import torch.nn.functional as F
    active = valid.bool()
    c.require(logits.shape == (1, 2, 256, 256) and bool(active.any()), "Invalid logits/valid domain")
    ce = F.cross_entropy(logits, target.long(), reduction="none")[active].mean()
    p = torch.softmax(logits, dim=1)[:, 1][active]; y = target.float()[active]
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return ce + dice, ce, dice

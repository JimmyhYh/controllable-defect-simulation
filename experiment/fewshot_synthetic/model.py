"""Lazy author-network wrapper; import alone never imports Torch."""
import importlib.util
from . import common as c


def configure(seed):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(seed):
    configure(seed)
    import torch
    from torch import nn
    path = c.ROOT / 'vendor/DRAEM/model_unet.py'
    expected = c.read_json(c.INPUT/'input_protocol.json')['source_identity']['vendor/DRAEM/model_unet.py']
    c.require(c.sha(path)==expected,'Author architecture identity differs')
    spec = importlib.util.spec_from_file_location('fewshot_pinned_draem_author',path)
    author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
    model = author.DiscriminativeSubNetwork(in_channels=3,out_channels=2,base_channels=16)
    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child,nn.BatchNorm2d):
                c.require(child.num_features % 8 == 0,'GN channels not divisible by8')
                setattr(module,name,nn.GroupNorm(8,child.num_features,eps=1e-5,affine=True))
            else:
                replace(child)
    replace(model)
    for module in model.modules():
        if isinstance(module,nn.Conv2d):
            nn.init.kaiming_normal_(module.weight,mode='fan_in',nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module,nn.GroupNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
    c.require(not any(isinstance(m,nn.BatchNorm2d) for m in model.modules()),'BN replacement incomplete')
    c.require(len(list(model.buffers()))==0,'Unexpected mutable buffers')
    return model


def state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name,t in model.state_dict().items():
        a=t.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(a.dtype).encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def loss_value(logits, target, valid):
    import torch
    from torch.nn import functional as F
    active=valid.bool()
    c.require(bool(active.any()),'Empty valid training domain')
    ce=F.cross_entropy(logits,target.long(),reduction='none')[active].mean()
    p=torch.softmax(logits,dim=1)[:,1][active]
    y=target.float()[active]
    dice=1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    return ce+dice,ce,dice

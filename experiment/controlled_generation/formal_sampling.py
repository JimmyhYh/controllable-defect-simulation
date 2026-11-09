"""CPU-only shared requests and pinned-author G0 / frozen controlled G1 inputs."""
import ast
import contextlib
import copy
import hashlib
import math
from functools import lru_cache
from types import SimpleNamespace

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.random as iar
import numpy as np

from .data import ROOT
from .sampling import generate_request


def shared_sequence(seed=20261030):
    indices = []
    for epoch in range(63):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([700001, seed, epoch])))
        indices.extend(rng.permutation(80).tolist())
    pair_backgrounds = np.array(indices[:5000], dtype=np.int16)
    return {'background_index':np.repeat(pair_backgrounds,2),
            'anomaly':np.tile(np.array([False,True]),5000),
            'anomaly_ordinal':np.array([-1 if k%2==0 else k//2 for k in range(10000)],np.int32)}


@lru_cache(maxsize=1)
def author_cpu_functions():
    """Compile exact pinned CPU-only AST bodies; do not import Dataset or torch."""
    tree = ast.parse((ROOT/'vendor/DRAEM/perlin.py').read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node,ast.FunctionDef)
                and node.name in ('lerp_np','rand_perlin_2d_np')]
    if len(selected)!=2:raise ValueError('Pinned author Perlin interface differs')
    loader = ast.parse((ROOT/'vendor/DRAEM/data_loader.py').read_text(encoding='utf-8'))
    cls = next(n for n in loader.body if isinstance(n,ast.ClassDef) and n.name=='MVTecDRAEMTrainDataset')
    init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignments = [n for n in init.body if isinstance(n,ast.Assign) and any(
                   isinstance(t,ast.Attribute) and t.attr in ('augmenters','rot') for t in n.targets)]
    if len(assignments)!=2:raise ValueError('Pinned author augmenter definitions differ')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='randAugmenter')
    namespace={'np':np,'math':math,'iaa':iaa}
    exec(compile(ast.Module(body=selected+[method],type_ignores=[]),'pinned_DRAEM_CPU_functions','exec'),namespace)
    return namespace, compile(ast.Module(body=assignments,type_ignores=[]),'pinned_DRAEM_augmenters','exec')


@contextlib.contextmanager
def isolated_cpu_random(seed):
    numpy_state=np.random.get_state()
    aug_rng=iar.get_global_rng();aug_state=copy.deepcopy(aug_rng.state)
    np.random.seed(seed);ia.seed(seed)
    try:yield
    finally:
        np.random.set_state(numpy_state)
        aug_rng.state=aug_state


def measured(rgb, image, roi):
    image=np.asarray(image,dtype=np.float32);roi=np.asarray(roi,dtype=bool)
    residual=np.max(np.abs(image-rgb),axis=-1)
    label=residual>np.float32(1/255)
    n=int(roi.sum());outside=float(residual[~roi].max(initial=0))
    coverage=float((label & roi).sum()/n) if n else 0.
    finite=bool(np.isfinite(image).all())
    valid=bool(finite and image.min()>=0 and image.max()<=1 and n and coverage>=.90
               and outside==0 and not np.any(label & ~roi))
    return {'image':image,'background':rgb,'M_roi':roi,'M_eff':label,
            'stats':{'valid':valid,'effective_coverage':coverage,'outside_ROI_max_float_residual':outside,
                     'roi_pixels':n,'positive_pixels':int(label.sum()),'finite':finite}}


def g0_request(rgb, root_seed, ordinal, texture_inventory):
    namespace,augmenter_code=author_cpu_functions()
    attempts=[]
    names=texture_inventory['names']
    for attempt in range(5):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([700002,root_seed,ordinal,attempt])))
        relative=names[int(rng.integers(len(names)))]
        component_seed=int(rng.integers(0,2**31-1))
        source_path=ROOT/'data/dtd/images'/relative
        if hashlib.sha256(source_path.read_bytes()).hexdigest()!=texture_inventory['sha256'][relative]:
            raise ValueError('DTD training-side texture identity changed: '+relative)
        source=cv2.imdecode(np.frombuffer(source_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if source is None:raise ValueError('Invalid DTD training image')
        source=cv2.resize(source,(256,256),interpolation=cv2.INTER_LINEAR)
        scales=tuple(2**int(rng.integers(0,6)) for _ in range(2))
        strength=1-float(rng.random())*.8
        with isolated_cpu_random(component_seed):
            generator=SimpleNamespace()
            scope=dict(namespace,self=generator)
            exec(augmenter_code,scope)
            augmenter=namespace['randAugmenter'](generator)
            texture_bgr=augmenter(image=source)
            noise=namespace['rand_perlin_2d_np']((256,256),scales)
            noise=generator.rot(image=noise)
        texture=np.ascontiguousarray(texture_bgr[:,:,::-1],dtype=np.float32)/np.float32(255)
        roi=noise>.5
        image=rgb+np.float32(strength)*roi[...,None]*(texture-rgb)
        result=measured(rgb,image,roi)
        attempts.append({'attempt':attempt+1,'texture':relative,'component_seed':component_seed,
                         'perlin_scales':list(scales),'lambda':strength,'stats':result['stats']})
        if result['stats']['valid']:return result,attempts
    return None,attempts


def training_sample(condition, cfg, sequence, step_index, backgrounds, texture_inventory):
    if condition not in ('G0','G1') or not 0<=step_index<10000:raise ValueError('Invalid formal request')
    rgb=backgrounds[int(sequence['background_index'][step_index])]
    if not sequence['anomaly'][step_index]:
        return {'image':rgb,'background':rgb,'M_eff':np.zeros((256,256),bool)}, {'anomaly':False,'attempts':[]}
    ordinal=int(sequence['anomaly_ordinal'][step_index])
    if condition=='G0':
        result,attempts=g0_request(rgb,cfg['seed'],ordinal,texture_inventory)
        kind,polarity='perlin_texture',None
    else:
        kind,polarity=cfg['G1_sampling']['formal_effective_anomaly_cycle'][ordinal%4]
        result,attempts=generate_request(rgb,kind,polarity,cfg['seed'],ordinal,cfg['G1_sampling'])
    info={'anomaly':True,'ordinal':ordinal,'kind':kind,'polarity':polarity,'attempts':attempts}
    if result is None:return None,info
    measurement=measured(rgb,result['image'],result['M_roi'])
    if not measurement['stats']['valid'] or not np.array_equal(measurement['M_eff'],result['M_eff']):
        raise ValueError('Generator and common actual-residual validity disagree')
    return measurement,info


def sample_fingerprint(sample):
    h=hashlib.sha256()
    h.update(np.ascontiguousarray(sample['image'],dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(sample['M_eff'],dtype=bool).tobytes())
    return h.hexdigest()

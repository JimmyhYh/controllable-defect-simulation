"""Paired, train-only CPU inputs. No Torch imports or model operations."""
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import common as c


def rng(seed, namespace):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026101911, seed, namespace])))


def load_window(row, y, flip_x=False, flip_y=False, real=False, inventory=None):
    """Native pixels; padding and target are distinct, including after flips."""
    if inventory is None:
        inventory = c.read_json(c.INPUT / 'export_inventory.json')
    # Inventory schema is fixed by day 1.
    hashes = inventory['files'] if 'files' in inventory else inventory
    path = c.DATA / row['image']
    c.require(c.sha(path) == hashes[row['image']], 'Training RGB hash differs')
    with Image.open(path) as im:
        c.require(im.size == (row['width'], row['height']) and im.mode in ('RGB','L'), 'Training RGB size/mode differs')
        rgb = np.asarray(im.convert('RGB'), dtype=np.float32) / np.float32(255)
    w, h = row['width'], row['height']
    c.require(2 <= w < 256 and y in c.d.window_starts(h), 'Invalid native window')
    image = np.pad(rgb[y:y+256], ((0,0),(0,256-w),(0,0)), mode='reflect')
    valid = np.zeros((256,256), bool); valid[:, :w] = True
    target = np.zeros((256,256), np.uint8)
    if real:
        mp = c.DATA / row['mask']
        c.require(c.sha(mp) == hashes[row['mask']], 'Training GT hash differs')
        with Image.open(mp) as im:
            a = np.asarray(im)
        c.require(a.shape == (h,w) and set(np.unique(a)).issubset({0,255}), 'GT is not original binary mask')
        target[:, :w] = (a[y:y+256] > 0).astype(np.uint8)
    for axis, enabled in ((1,flip_x),(0,flip_y)):
        if enabled:
            image, target, valid = (np.flip(a, axis=axis) for a in (image,target,valid))
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(target), np.ascontiguousarray(valid)


def sequences(split, seed):
    real_rows = split['R_real5']; normals = split['N_train1885']
    geometry = c.read_json(c.INPUT / 'review_geometry.json')['samples']
    positives = {r['id']: [w['y'] for w in r['windows'] if w['positive_pixels'] > 0]
                 for r in geometry if r['role'] == 'R_real5'}
    real_rng, bg_rng, aug_rng = (rng(seed, n) for n in (1,2,3))
    real_order = [int(x) for _ in range(1000) for x in real_rng.permutation(5)]
    rows = []; r_index = 0; synth_index = 0
    for step in range(1,10001):
        slot = (step-1) % 4
        if slot in (0,2):
            row = real_rows[real_order[r_index]]; r_index += 1
            choices = positives[row['id']]
            y = int(real_rng.choice(choices)); role = 'real'
        else:
            row = normals[int(bg_rng.integers(len(normals)))]
            y = int(bg_rng.choice(c.d.window_starts(row['height'])))
            role = 'normal' if slot == 1 else 'intervention'
        item = {'step':step, 'role':role, 'id':row['id'], 'y':y,
                'flip_x':bool(aug_rng.random() < .5), 'flip_y':bool(aug_rng.random() < .5)}
        if role == 'intervention':
            item['synthetic_ordinal'] = synth_index; synth_index += 1
        rows.append(item)
    c.require(Counter(r['id'] for r in rows if r['role']=='real') == Counter({r['id']:1000 for r in real_rows}), 'Real exposure differs')
    return rows


def spec_for(valid, seed, ordinal, attempt):
    gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026101910,seed,ordinal,attempt])))
    category = ordinal % 4
    angle = float(gen.uniform(0,180)); theta = np.deg2rad(angle)
    if category < 2:
        axes = gen.uniform(8,24,size=2)
        extent = np.sqrt([(axes[0]*np.cos(theta))**2+(axes[1]*np.sin(theta))**2,
                          (axes[0]*np.sin(theta))**2+(axes[1]*np.cos(theta))**2])
        params = {'kind':'mura_like','shape':'ellipse','axes':axes.tolist()}
    else:
        length, width = float(gen.uniform(24,96)), float(gen.uniform(2,6))
        extent = np.abs([np.cos(theta),np.sin(theta)])*length/2 + width/2
        params = {'kind':'line_like','shape':'line','length':length,'width':width}
    yy, xx = np.nonzero(valid)
    lo, hi = np.array([xx.min(),yy.min()])+extent, np.array([xx.max(),yy.max()])-extent
    c.require(np.all(hi > lo), 'Requested geometry cannot fit fixed valid domain')
    center = gen.uniform(lo,hi)
    return {**params, 'units':'px256','center':center.tolist(),'angle_deg':angle,
            'amplitude': .04 if category % 2 == 0 else -.04}


def save_pack(path, image, target, valid):
    c.require(not path.exists(), 'Never overwrite an existing input')
    np.savez(path, image=image.astype(np.float32), target=target.astype(np.uint8), valid=valid.astype(np.uint8))


def config(identity):
    return {'schema':'fewshot-paired-v1', 'input_identity':identity, 'implementation':c.identities(),
            'environment':c.environment(), 'formal_seeds':list(c.SEEDS),'probe_seed':c.PROBE_SEED,
            'formal_steps':10000,'probe_steps':1000,'real_count':5,'normal_count':1885,
            'slots':['real','normal','real','intervention'], 'synthetic_per_seed':2500,
            'synthetic_types':['bright_mura','dark_mura','bright_line','dark_line'],
            'amplitude':.04,'attempts':5,'effective_threshold':float(np.float32(1/255)),
            'coverage_min':.9, 'window':256, 'stride':128,'right_padding':'reflect; excluded from loss and synthesis',
            'sequence_namespace':[2026101911,'training_seed','1=real,2=background,3=augmentation'],
            'synthetic_namespace':[2026101910,'training_seed','ordinal_zero_based','attempt_zero_based_0_to_4'],
            'probe_shuffle_namespace':[2026101912,c.PROBE_SEED,'epoch_zero_based'],
            'model':'author DiscriminativeSubNetwork(3,2,base_channels=16); replace all BN by GN(8,C,eps=1e-5)',
            'initialization':'Conv Kaiming normal fan_in relu; bias0; GN weight1 bias0; no old weights',
            'loss':'mean valid-domain unweighted CE + (1-(2*sum(p*y)+1)/(sum(p)+sum(y)+1))',
            'optimizer':{'name':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0},
            'formal_lr':[[1,8000,1e-4],[8001,9000,2e-5],[9001,10000,4e-6]],
            'probe_lr':[[1,800,1e-4],[801,900,2e-5],[901,1000,4e-6]],
            'FP32':True,'batch':1,'workers':0,'TF32':False,'deterministic':True,'probe_gate_AP':.8,
            'checkpoint_every':{'probe':200,'formal':1000},'min_training_RAM_GiB':3,'min_free_VRAM_GiB':6,
            'inference':{'overlap':'equal mean','score':'max stitched native probability','q':.95,'method':'linear','decision':'strict >'},
            'metrics':{'N_test_native':146411528,'float32_histogram_bins':0x3f800001,'chunk':1048576,'minimum_free_disk_GiB':80}}


def prepare_pairs():
    split, identity = c.source_metadata()
    c.guard(split, cpu=True)
    if (c.PREP / 'complete.json').exists():
        return verify_pairs(split, identity)
    c.require(not c.PREP.exists(), 'Unknown/incomplete preparation; preserve and investigate')
    resources = c.resource()
    from controlled_generation.render import render
    started = time.monotonic()
    c.PREP.mkdir(parents=True)
    c.save(c.PREP/'transaction.json', {'input_identity':identity,'implementation':c.identities(),'resources':resources})
    cfg = config(identity); c.save(c.PREP/'common_config.json', cfg)
    inventory = c.read_json(c.INPUT/'export_inventory.json')
    by_id = {r['id']: r for role in ('R_real5','N_train1885') for r in split[role]}
    all_stats = []; render_calls = 0
    for seed in c.SEEDS:
        folder = c.PREP/str(seed); folder.mkdir()
        (folder/'synthetic').mkdir()
        rows = sequences(split,seed)
        c.save(folder/'sequence.json',rows)
        for row in rows:
            if row['role'] != 'intervention':
                continue
            ordinal = row['synthetic_ordinal']
            image, _, valid = load_window(by_id[row['id']],row['y'],row['flip_x'],row['flip_y'],inventory=inventory)
            attempts = []; accepted = None
            for attempt in range(5):
                spec = spec_for(valid,seed,ordinal,attempt)
                product = render(image,spec,seed=seed); render_calls += 1
                residual = np.max(np.abs(product['image']-image),axis=-1)
                target = residual > np.float32(1/255)
                outside = float(residual[~valid].max(initial=0))
                valid_ok = bool(product['stats']['valid'] and not np.any(product['M_roi'] & ~valid)
                                and target.any() and product['stats']['effective_coverage'] >= .9 and outside == 0
                                and np.array_equal(target,product['M_eff']))
                attempts.append({'attempt':attempt,'spec':spec,'accepted':valid_ok,'stats':product['stats'],
                                 'outside_valid_max_residual':outside})
                if valid_ok:
                    accepted = product; break
            record = {'seed':seed,'ordinal':ordinal,'sequence_step':row['step'],'type':cfg['synthetic_types'][ordinal%4],
                      'background_request':row,'attempts':attempts,'valid':accepted is not None}
            if accepted is None:
                c.save(c.PREP/'invalid_input.json',record)
                raise RuntimeError('Five geometry attempts exhausted; no background/seed/amplitude substitution')
            path = folder/'synthetic'/f'{ordinal:04d}.npz'
            save_pack(path,accepted['image'],target,valid)
            record.update(file=path.relative_to(c.PREP).as_posix(),sha256=c.sha(path),
                          positive_pixels=int(target.sum()),valid_pixels=int(valid.sum()))
            all_stats.append(record)
            if (ordinal+1)%250 == 0:
                print({'seed':seed,'valid_synthetic':ordinal+1,'of':2500,'elapsed_seconds':time.monotonic()-started},flush=True)
                c.resource()
    c.save(c.PREP/'synthetic_manifest.json',all_stats)
    probe = []
    geometry = c.read_json(c.INPUT/'review_geometry.json')['samples']
    for row in split['R_real5']:
        g = next(g for g in geometry if g['id']==row['id'])
        y = next(w['y'] for w in g['windows'] if w['positive_pixels']>0)
        probe.append({'kind':'real','id':row['id'],'y':y,'flip_x':False,'flip_y':False})
    for ordinal in range(3):
        probe.append({'kind':'synthetic','file':f'{c.SEEDS[0]}/synthetic/{ordinal:04d}.npz','ordinal':ordinal})
    scope = c.read_json(c.INPUT/'input_protocol.json')['visual_scope']['N16']
    for sample_id in sorted(scope)[:4]:
        probe.append({'kind':'normal','id':sample_id,'y':0,'flip_x':False,'flip_y':False})
    (c.PREP/'probe_inputs').mkdir()
    for i,item in enumerate(probe):
        if item['kind']=='synthetic':
            with np.load(c.PREP/item['file'],allow_pickle=False) as a:
                image,target,valid=(a[n] for n in ('image','target','valid'))
        else:
            image,target,valid=load_window(by_id[item['id']],item['y'],real=item['kind']=='real',inventory=inventory)
        path=c.PREP/'probe_inputs'/f'{i:02d}.npz'
        save_pack(path,image,target,valid)
        item.update(file=path.relative_to(c.PREP).as_posix(),sha256=c.sha(path),positive_pixels=int(target.sum()),valid_pixels=int(valid.sum()))
    c.save(c.PREP/'probe.json',probe)
    boundary = {k:{str(n):c.lr_at(n,probe=p) for n in points} for k,p,points in
                [('probe',True,(1,800,801,900,901,1000)),('formal',False,(1,8000,8001,9000,9001,10000))]}
    summary={'effective_synthetic':len(all_stats),'render_calls':render_calls,'retry_calls':render_calls-7500,
             'type_counts':{str(seed):dict(Counter(r['type'] for r in all_stats if r['seed']==seed)) for seed in c.SEEDS},
             'probe_inputs':len(probe),'LR_boundaries':boundary,'resources_at_start':resources,
             'resources_at_finish':c.resource(),'elapsed_seconds':time.monotonic()-started,
             'no_Torch_import': 'torch' not in __import__('sys').modules,'new_model_operations':0,
             'Q_E_U_image_or_label_reads':0,'training_started':False}
    c.save(c.PREP/'preparation_summary.json',summary)
    c.seal(c.PREP,{'input_identity':identity,'implementation':c.identities(),'environment':c.environment()})
    return verify_pairs(split,identity)


def verify_pairs(split=None, identity=None):
    if split is None:
        split, identity = c.source_metadata()
        c.guard(split,cpu=True)
    complete = c.verify_manifest(c.PREP)
    c.require(complete['input_identity']==identity and complete['implementation']==c.identities(), 'Preparation identity differs')
    c.require(complete['environment']==c.environment(),'Interpreter/dependencies differ')
    stats=c.read_json(c.PREP/'synthetic_manifest.json'); probe=c.read_json(c.PREP/'probe.json')
    c.require(len(stats)==7500 and len(probe)==12 and all(r['valid'] for r in stats),'Input population incomplete')
    for seed in c.SEEDS:
        rows=c.read_json(c.PREP/str(seed)/'sequence.json')
        c.require(rows==sequences(split,seed),'Common paired sequence differs')
        c.require(Counter(r['type'] for r in stats if r['seed']==seed)==Counter(dict.fromkeys(config(identity)['synthetic_types'],625)), 'Synthetic type counts differ')
    return {'preparation_verified':True,'effective_synthetic':7500,'probe_inputs':12,'cache_reuse':True,
            'training_started':False,'model_operations':0,'summary':c.read_json(c.PREP/'preparation_summary.json')}

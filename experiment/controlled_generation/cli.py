"""CPU preparation and frozen-model demos; only explicit user-started probe training commands update parameters."""
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import OUT, ROOT, SPLIT, code_identity, digest, load_split, normal_rgb, prepare, read_json, save_json
from .render import render


def requests():
    cases = []
    defaults = {'units': 'px256', 'amplitude': .04, 'angle_deg': 0}
    def pair(kind, factor, first, second, background, seed):
        for variant, supplied in enumerate((first, second)):
            spec = {**defaults, 'kind': kind, **supplied}
            cases.append({'id': f'{kind}_{len(cases)+1:02d}', 'pair_id': len(cases)//2 + 1,
                          'changed_factor': factor, 'variant': variant, 'background_index': background,
                          'seed': seed, 'spec': spec})
    ellipse = {'shape': 'ellipse', 'center': [82, 100], 'axes': [28, 18]}
    pair('mura_like', 'center_x', ellipse, {**ellipse, 'center': [172,100]}, 0, 101)
    ellipse = {'shape': 'ellipse', 'center': [128,128], 'axes': [18,12]}
    pair('mura_like', 'semiaxes', ellipse, {**ellipse, 'axes': [38,26]}, 1, 102)
    ellipse = {'shape': 'ellipse', 'center': [128,128], 'axes': [38,16]}
    pair('mura_like', 'angle', ellipse, {**ellipse, 'angle_deg': 55}, 2, 103)
    circle = {'shape': 'circle', 'center': [150,125], 'radius': 18, 'amplitude': -.04}
    pair('mura_like', 'radius', circle, {**circle, 'radius': 32}, 3, 104)
    blob = {'shape': 'anostyler_blob', 'center': [128,128], 'size': [42,28], 'amplitude': -.04}
    pair('mura_like', 'bbox_size', blob, {**blob, 'size': [74,48]}, 0, 105)
    poly = {'shape': 'polygon', 'vertices': [[12,55],[38,28],[62,48],[45,77]]}
    pair('mura_like', 'vertex_translation_x', poly,
         {**poly, 'vertices': [[x+55,y] for x,y in poly['vertices']]}, 1, 106)
    line = {'shape': 'line', 'center': [85,120], 'length': 90, 'width': 4, 'angle_deg': 30}
    pair('line_like', 'center_x', line, {**line, 'center': [165,120]}, 0, 201)
    line = {'shape': 'line', 'center': [128,128], 'length': 50, 'width': 4, 'angle_deg': 25}
    pair('line_like', 'length', line, {**line, 'length': 130}, 1, 202)
    line = {'shape': 'line', 'center': [128,128], 'length': 110, 'width': 3, 'angle_deg': 25}
    pair('line_like', 'width', line, {**line, 'width': 7}, 2, 203)
    line = {'shape': 'line', 'center': [128,128], 'length': 110, 'width': 4, 'angle_deg': 0, 'amplitude': -.04}
    pair('line_like', 'angle', line, {**line, 'angle_deg': 60}, 3, 204)
    poly = {'shape': 'polyline', 'points': [[65,95],[110,95],[155,95]], 'width': 4, 'amplitude': -.04}
    pair('line_like', 'middle_vertex_y', poly, {**poly, 'points': [[65,95],[110,120],[155,95]]}, 0, 205)
    line = {'shape': 'line', 'center': [128,18], 'length': 140, 'width': 4}
    pair('line_like', 'center_y_near_edge', line, {**line, 'center': [128,44]}, 1, 206)
    return {'schema': 'controlled-24-predeclared-requests-v1', 'total_requests': 24, 'pairs': 12,
            'types_each': 12, 'background_rule': 'first four N_gen paths, preassigned pair indices; no visual selection',
            'same_background_and_seed_within_pair': True, 'cases': cases}


def save_sample(folder, result, case):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = folder / 'arrays.npz'
    with arrays.open('wb') as stream:
        np.savez_compressed(stream, background=result['background'], image=result['image'], alpha=result['alpha'],
                            M_roi=result['M_roi'], M_eff=result['M_eff'], M_eff_png=result['M_eff_png'], residual=result['residual'])
    images = {'background.png': result['background_png'], 'generated.png': result['image_png'],
              'requested_mask.png': result['M_roi'].astype(np.uint8)*255,
              'effective_mask.png': result['M_eff'].astype(np.uint8)*255,
              'effective_mask_png.png': result['M_eff_png'].astype(np.uint8)*255}
    for name, pixels in images.items():
        Image.fromarray(pixels).save(folder / name)
    files = {p.name: digest(p) for p in [arrays] + [folder / n for n in images]}
    record = {'request': case, 'stats': result['stats'], 'files_sha256': files}
    save_json(folder / 'result.json', record)
    return record


def invalid_requests(rgb):
    base = {'kind':'mura_like','shape':'circle','units':'px256','center':[128,128],'radius':20,'amplitude':.04}
    variants = [('outside_frame',{**base,'center':[2,2]}),
                ('zero_radius',{**base,'radius':0}), ('wrong_units',{**base,'units':'normalized'}),
                ('thin_degenerate_line',{**base,'shape':'line','length':60,'width':1,'angle_deg':0}),
                ('self_intersection',{**base,'shape':'polygon','vertices':[[80,80],[120,120],[80,120],[120,80]]}),
                ('zero_amplitude',{**base,'amplitude':0})]
    observations = []
    for name, spec in variants:
        try:
            result = render(rgb,spec)
            rejected = not result['stats']['valid']; reason=result['stats']['invalid_reasons']
        except ValueError as error:
            rejected=True;reason=[str(error)]
        observations.append({'case':name,'rejected':rejected,'reason':reason})
    for name, background, amplitude in [('white_positive_clip',np.ones_like(rgb),.04),
                                        ('black_negative_clip',np.zeros_like(rgb),-.04)]:
        result = render(background,{**base,'amplitude':amplitude})
        observations.append({'case':name,'rejected':not result['stats']['valid'],
                             'reason':result['stats']['invalid_reasons'], 'coverage':result['stats']['effective_coverage']})
    return observations


def demo():
    split = load_split(); planned=requests()
    request_path = OUT / 'trial_requests.json'
    if request_path.exists() and read_json(request_path) != planned:
        raise ValueError('Predeclared request protocol differs')
    save_json(request_path, planned)  # Freeze before generating or inspecting detector responses.
    signature = {'requests_sha256':digest(request_path), 'split_sha256':digest(SPLIT),
                 'generation_protocol_sha256':digest(OUT/'generation_protocol.json'), 'implementation':code_identity()}
    complete_path = OUT / 'trial_results.json'
    if complete_path.exists():
        prior=read_json(complete_path)
        old=prior['signature']
        core=('data.py','geometry.py','render.py','__init__.py')
        same_core=all(old['implementation'].get(name)==sha for name,sha in signature['implementation'].items()
                      if Path(name).name in core)
        if {k:v for k,v in old.items() if k!='implementation'} != {k:v for k,v in signature.items() if k!='implementation'} or not same_core:
            raise ValueError('Existing trial source/config differs; preserve prior artifacts')
        audit();return prior
    start=time.perf_counter(); records=[]
    for case in planned['cases']:
        source=split['N_gen'][case['background_index']]
        background=normal_rgb(split,source)
        case={**case,'background_path':source,'source_sha256':split['source_sha256'][source]}
        result=render(background,case['spec'],case['seed'])
        records.append(save_sample(OUT/'trial'/case['id'],result,case))
    invalid=invalid_requests(normal_rgb(split,split['N_gen'][0]))
    good=sum(r['stats']['valid'] for r in records)
    summary={'signature':signature,'requests':24,'pairs':12,'unique_background_images':4,
             'valid_requests':good,'invalid_requests':24-good,'valid_by_type':{
                 kind:sum(r['stats']['valid'] for r in records if r['stats']['kind']==kind)
                 for kind in ('mura_like','line_like')}, 'actual_validity_observations':invalid,
             'rejected_invalid_conditions':sum(r['rejected'] for r in invalid),
             'residual_range_all_ROI': [min(r['stats']['roi_residual_min_median_max'][0] for r in records),
                                        max(r['stats']['roi_residual_min_median_max'][2] for r in records)],
             'minimum_effective_coverage':min(r['stats']['effective_coverage'] for r in records),
             'minimum_png_effective_coverage':min(r['stats']['png_effective_coverage'] for r in records),
             'max_outside_ROI_float_residual':max(r['stats']['outside_ROI_max_float_residual'] for r in records),
             'max_outside_ROI_png_codes':max(r['stats']['outside_ROI_max_png_codes'] for r in records),
             'cpu_generation_and_save_seconds':time.perf_counter()-start,
             'records':records,'training_started':False,'new_independent_real_evaluation':False,
             'reproducibility_scope':'same input/spec/seed and pinned implementation; CPU generator only',
             'scientific_scope':'control and annotation audit; no detector improvement or physical realism claim'}
    save_json(complete_path,summary)
    audit()
    return summary


def audit():
    split=load_split();summary=read_json(OUT/'trial_results.json'); failures=[]
    signature=summary['signature']
    core=('data.py','geometry.py','render.py','__init__.py')
    if any(signature['implementation'].get(name)!=sha for name,sha in code_identity().items() if Path(name).name in core):
        raise ValueError('Geometry/render/input code has changed since generation')
    if (digest(OUT/'trial_requests.json')!=signature['requests_sha256'] or digest(SPLIT)!=signature['split_sha256']
            or digest(OUT/'generation_protocol.json')!=signature['generation_protocol_sha256']):
        raise ValueError('Saved request/data protocol changed')
    for entry in summary['records']:
        request=entry['request']; folder=OUT/'trial'/request['id']
        if read_json(folder/'result.json') != entry:
            failures.append(request['id']+': sample scientific record differs')
        for name,sha in entry['files_sha256'].items():
            if digest(folder/name) != sha:
                failures.append(request['id']+': artifact hash differs')
        with np.load(folder/'arrays.npz',allow_pickle=False) as stored:
            before=stored['background']; after=stored['image'];roi=stored['M_roi']
            residual=np.max(np.abs(after-before),axis=-1)
            mask=residual > np.float32(1/255)
            if not np.array_equal(mask,stored['M_eff']) or not np.array_equal(residual,stored['residual']):
                failures.append(request['id']+': saved float label/residual mismatch')
            if np.any(residual[~roi]!=0):
                failures.append(request['id']+': changed pixel outside ROI')
            rerender=render(normal_rgb(split,request['background_path']),request['spec'],request['seed'])
            if not np.array_equal(after,rerender['image']) or not np.array_equal(roi,rerender['M_roi']):
                failures.append(request['id']+': deterministic rerender differs')
            if rerender['stats'] != entry['stats']:
                failures.append(request['id']+': reported statistics differ from actual rerender')
        a=np.asarray(Image.open(folder/'background.png')).astype(np.int16)
        b=np.asarray(Image.open(folder/'generated.png')).astype(np.int16)
        effective_png=np.max(np.abs(b-a),axis=-1)>1
        saved=np.asarray(Image.open(folder/'effective_mask_png.png'))>0
        if not np.array_equal(effective_png,saved) or np.any(np.max(np.abs(b-a),axis=-1)[~roi]!=0):
            failures.append(request['id']+': PNG residual or label mismatch')
    result={'trial_results_sha256':digest(OUT/'trial_results.json'),
            'auditor_sha256':digest(Path(__file__)),
            'requests_verified':len(summary['records']), 'failures':failures,
            'same_input_rerender_equal':not failures, 'sample_pairs_not_independent_images':True,
            'training_started':False}
    save_json(OUT/'artifact_audit.json',result)
    if failures:
        raise ValueError('; '.join(failures))
    print('AUDIT',result,flush=True)
    return result


def generate(spec_file,background,output_dir,seed):
    split=load_split()
    spec=read_json(spec_file)
    result=render(normal_rgb(split,background),spec,seed)
    output=Path(output_dir).resolve()
    if not output.is_relative_to(OUT.resolve()) or output.exists():
        raise ValueError('Use a fresh output directory under results/controlled_generation')
    record=save_sample(output,result,{'spec':spec,'seed':seed,'background_path':background})
    print('GENERATED',record['stats'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','demo','audit','figures','predict','prepare-training','verify-probe','probe-status',
                 'prepare-batchstats-probe','verify-batchstats-probe','batchstats-probe-status','diagnose-learning',
                 'prepare-lr-decay-probe','verify-lr-decay-probe','lr-decay-probe-status',
                 'formal-prepare','verify-formal','calibration-prepare','verify-calibration-preparation',
                 'holdout-prepare','verify-holdout-preparation','holdout-generate','verify-holdout-data',
                 'holdout-score','verify-holdout-scores','holdout-metrics','verify-holdout-metrics'):
        sub.add_parser(name)
    train_parser=sub.add_parser('probe-train',help='Explicit user-started1000-step diagnostic; never called by prepare')
    train_parser.add_argument('--seed',type=int,default=20261029)
    batchstats_parser=sub.add_parser('batchstats-probe-train',help='Independent user-started1000-step stateless-BN probe')
    batchstats_parser.add_argument('--seed',type=int,default=20261029)
    lr_parser=sub.add_parser('lr-decay-probe-train',help='Explicit user-started independent1000-step LR-decay probe')
    lr_parser.add_argument('--seed',type=int,default=20261029)
    formal_status=sub.add_parser('formal-status',help='Read-only owned formal result/checkpoint identity')
    formal_status.add_argument('--condition',choices=('G0','G1'),required=True)
    formal_train=sub.add_parser('formal-train',help='Explicit user-started independent10000-step formal training')
    formal_train.add_argument('--condition',choices=('G0','G1'),required=True)
    formal_train.add_argument('--seed',type=int,default=20261030)
    calibration_parser=sub.add_parser('calibration-run',help='Explicit Q_cal scoring; never called by preparation')
    calibration_parser.add_argument('--condition',choices=('G0','G1'),required=True)
    generate_parser=sub.add_parser('generate')
    generate_parser.add_argument('--spec',required=True)
    generate_parser.add_argument('--background',required=True,help='Relative N_gen/S_bg normal path')
    generate_parser.add_argument('--output-dir',required=True)
    generate_parser.add_argument('--seed',type=int,default=20261014)
    args=parser.parse_args()
    if args.command=='prepare':
        split=prepare();print('PREPARED',split['counts'],split['qualified_RGB_groups'])
    elif args.command=='demo':
        result=demo();print('DEMO', {key:result[key] for key in ['requests','valid_requests','valid_by_type','minimum_effective_coverage','max_outside_ROI_float_residual','cpu_generation_and_save_seconds']})
    elif args.command=='audit':audit()
    elif args.command=='prepare-training':
        from .preparation import prepare_probe
        prepare_probe()
    elif args.command=='verify-probe':
        from .preparation import verify_probe
        verify_probe()
    elif args.command=='probe-status':
        from .training import completed_status
        completed_status()
    elif args.command=='probe-train':
        from .training import train_probe
        train_probe(args.seed)
    elif args.command=='prepare-batchstats-probe':
        from .batchstats_preparation import prepare_probe
        prepare_probe()
    elif args.command=='verify-batchstats-probe':
        from .batchstats_preparation import verify_probe
        verify_probe()
    elif args.command=='batchstats-probe-status':
        from .batchstats_training import completed_status
        completed_status()
    elif args.command=='batchstats-probe-train':
        from .batchstats_training import train_probe
        train_probe(args.seed)
    elif args.command=='diagnose-learning':
        from .probe_learning_diagnostics import diagnose
        diagnose()
    elif args.command=='prepare-lr-decay-probe':
        from .probe_lr_decay_preparation import prepare_lr_decay_probe
        prepare_lr_decay_probe()
    elif args.command=='verify-lr-decay-probe':
        from .probe_lr_decay_preparation import verify_lr_decay_probe
        verify_lr_decay_probe()
    elif args.command=='lr-decay-probe-status':
        from .probe_lr_decay_training import completed_lr_decay_status
        completed_lr_decay_status()
    elif args.command=='lr-decay-probe-train':
        from .probe_lr_decay_training import train_lr_decay_probe
        train_lr_decay_probe(args.seed)
    elif args.command=='formal-prepare':
        from .formal_preparation import prepare_formal
        prepare_formal()
    elif args.command=='verify-formal':
        from .formal_preparation import verify_formal_preparation
        verify_formal_preparation()
    elif args.command=='formal-status':
        from .formal_training import completed_formal_status
        completed_formal_status(args.condition)
    elif args.command=='formal-train':
        from .formal_training import train_formal
        train_formal(args.condition,args.seed)
    elif args.command=='calibration-prepare':
        from .calibration import prepare_calibration
        prepare_calibration()
    elif args.command=='verify-calibration-preparation':
        from .calibration import verify_preparation
        verify_preparation()
    elif args.command=='calibration-run':
        from .calibration import run_calibration
        run_calibration(args.condition)
    elif args.command=='holdout-prepare':
        from .holdout_preparation import prepare_holdout
        prepare_holdout()
    elif args.command=='verify-holdout-preparation':
        from .holdout_preparation import verify_preparation
        verify_preparation()
    elif args.command=='holdout-generate':
        from .holdout_data import generate_holdout
        generate_holdout()
    elif args.command=='verify-holdout-data':
        from .holdout_data import verify_generated
        verify_generated()
    elif args.command=='holdout-score':
        from .holdout_inference import run_scoring
        run_scoring()
    elif args.command=='verify-holdout-scores':
        from .holdout_inference import verify_scores
        verify_scores()
    elif args.command=='holdout-metrics':
        from .holdout_metrics import run_metrics
        run_metrics()
    elif args.command=='verify-holdout-metrics':
        from .holdout_metrics import verify_metrics
        verify_metrics()
    elif args.command=='generate':generate(args.spec,args.background,args.output_dir,args.seed)
    elif args.command=='figures':
        from .figures import draw
        draw()
    elif args.command=='predict':
        from .evaluation import predict_trial
        predict_trial()


if __name__=='__main__':
    main()

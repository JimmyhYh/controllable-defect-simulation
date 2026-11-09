"""Day2 prepare/probe CLI. Formal training and evaluation are later tasks."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from localization_data import ROOT, OUT, SPLIT, save, prepare, digest

def preparation():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if SPLIT.exists() and protocol['scientific_state']['split_frozen']:
        stored=json.loads(SPLIT.read_text(encoding='utf-8'))
        split=prepare(stored['N_cap'])
        if digest(SPLIT) != protocol['split_sha256']:
            raise ValueError('Frozen split identity differs')
        return split
    from localization_train import available_ram, process_ram
    free = available_ram()
    resident = process_ram()['resident_GiB']
    estimates = {cap: 1.0 + 2 * cap * 1024 * 1024 * 4 / 2**30 for cap in (256,160,80)}
    cap = next((n for n in (256,160,80) if max(0, estimates[n]-resident) <= .8 * free), None)
    if cap is None:
        raise RuntimeError(f'Insufficient RAM for conservative PatchCore resource estimate: {free:.3f} GiB available; minimum estimate {estimates[80]:.3f} GiB')
    split = prepare(cap)
    save(OUT/'prepare_summary.json', {'available_ram_GiB': free, 'current_process_resident_GiB':resident, 'PatchCore_memory_estimate_GiB': estimates,
          'estimate_rule': '1GiB fixed peak reserve + two feature copies (author list/concatenation); subtract current resident to compare incremental demand with 80% available RAM; estimate not measured bank peak',
          'N_cap':cap, 'counts':split['counts'], 'split_sha256':digest(SPLIT), 'group_seed':20261012,
          'model_inference':False, 'bank_built':False})
    return split

def check_synthesis(split):
    from localization_synthesis import PairStream, blend, residual
    stream = PairStream(split, 20261013, first_eight=True)
    samples=[]
    for _ in range(320):
        sample=stream.next(force_anomaly=True)
        if sample['active']:
            samples.append(sample)
        if len(samples)==32:
            break
    if len(samples)!=32:
        raise RuntimeError('Could not obtain 32 effective masks')
    candidates=[]; chosen=None
    for weak in (.1,.15,.2):
        rows=[]
        for i,sample in enumerate(samples):
            a=blend(sample,sample['lambda_A']);b=blend(sample,weak)
            ar=residual(a,sample);br=residual(b,sample)
            rows.append({'index':i, **sample['metadata'], 'lambda_A':sample['lambda_A'], 'lambda_weak':weak,
                'mask_sha256':hashlib.sha256(sample['mask'].tobytes()).hexdigest(), 'A':ar, 'weak':br,
                'passes':bool(br['finite'] and br['outside_max_abs']<=1e-6 and br['effective_pixel_fraction']>=.9)})
        number=sum(r['passes'] for r in rows)
        candidates.append({'lambda':weak,'passing_pairs':number,'pairs':rows})
        if number>=30 and all(r['weak']['finite'] and r['weak']['outside_max_abs']<=1e-6 for r in rows):
            chosen=weak; break
    # All 32 samples remain visible, regardless of success; four sheets, eight rows each.
    display_weak=chosen if chosen is not None else .2
    for page in range(4):
        canvas=Image.new('RGB',(1024,8*280),'white');draw=ImageDraw.Draw(canvas)
        for row,sample in enumerate(samples[page*8:(page+1)*8]):
            tiles=[sample['background'],blend(sample,sample['lambda_A']),blend(sample,display_weak),np.repeat(sample['mask'][...,None],3,axis=2)]
            for col,tile in enumerate(tiles):
                canvas.paste(Image.fromarray(np.rint(tile*255).astype(np.uint8)),(col*256,row*280))
            draw.text((4,row*280+257),f'{page*8+row:02d} | N background | A lambda={sample["lambda_A"]:.3f} | weak={display_weak} | mask',fill='black')
        canvas.save(OUT/f'training_pairs_{page+1}.png')
    save(OUT/'synthesis_check.json',{'seed':20261013,'denominator':32,'chosen_lambda':chosen,
         'candidates':candidates,'normal_background_paths':split['N'][:8],'held_out_images_read':False,
         'diagnostic_forced_anomaly':True,'background_training_rotation_probability':.3,
         'label_rule':'shared binary Perlin >0.5; effective change threshold strictly >1/255',
         'visualization_only_quantized':True})
    if chosen is None:
        raise RuntimeError('All prespecified lambda candidates failed validity gate; do not train B')
    return chosen

def probe():
    from localization_train import run_probe
    split=json.loads(SPLIT.read_text(encoding='utf-8'))
    if (OUT/'probe_results.json').exists():
        cached=json.loads((OUT/'probe_results.json').read_text(encoding='utf-8'))
        check=json.loads((OUT/'synthesis_check.json').read_text(encoding='utf-8'))
        if not cached['complete'] or cached['background_pool'] != split['N'][:8] or check['normal_background_paths'] != split['N'][:8]:
            raise ValueError('Probe cache incomplete or training backgrounds differ; no automatic rerun')
        if 'split_sha256' in cached and cached['split_sha256'] != digest(SPLIT):
            raise ValueError('Frozen probe split differs')
        weak=check['chosen_lambda'];results=cached['results']
        print('Using complete train-only resource probe cache; no GPU rerun',flush=True)
    else:
        weak=check_synthesis(split)
        results=[]
        for condition in ('A','B'):
            print(f'PROBE {condition} fp32: 10 warmup + 30 timed steps',flush=True)
            results.append(run_probe(split,weak,condition))
            save(OUT/'probe_results.json',{'results':results,'complete':len(results)==2,
                'split_sha256':digest(SPLIT),'background_pool':split['N'][:8]})
    if weak is None or len(results)!=2 or any(not r['weights_changed'] or not np.isfinite(r['losses']).all() for r in results):
        raise ValueError('Invalid resource probe evidence')
    if results[0]['initial_first_conv_sha256']!=results[1]['initial_first_conv_sha256']:
        raise RuntimeError('Pair initialization mismatch')
    t=max(r['median_step_seconds'] for r in results)
    steps=next((s for s in (60000,30000,10000) if 2*s*t*1.25<=8*3600),None)
    if steps is None:
        raise RuntimeError('No prespecified training budget meets 8-hour rule')
    pair_seconds=2*steps*t*1.25
    seeds=[20261020,20261021,20261022] if 3*pair_seconds<=24*3600 else [20261020]
    config={'input_size':256,'batch_size':1,'num_workers':0,'precision':'fp32','learning_rate':1e-4,
        'weight_decay':0,'steps':steps,'milestones':[int(.8*steps),int(.9*steps)],'gamma':.2,
        'paired_seeds':seeds,'data_seed_offset':100000,'lambda_weak':weak,'replacement_fraction':.5,
        'N_cap':split['N_cap'],'split_sha256':digest(SPLIT),'slower_median_step_seconds':t,
        'estimated_pair_hours':pair_seconds/3600,'estimated_all_pair_hours':len(seeds)*pair_seconds/3600,
        'models':'author ReconstructiveSubNetwork base128 + DiscriminativeSubNetwork base64',
        'loss':'MSE + SSIM + Focal, author definitions','checkpoint_rule':'final fixed step; probe weights discarded',
        'image_score_DRAEM':'max anomaly softmax probability','image_threshold':'Vnew q0.95 linear; score>threshold',
        'pixel_evaluation':'all final test pixels at256; GT nearest; maps bilinear; global noninterpolated AP',
        'training_background_rotation':.3,'evaluation_background_augmentation':False,
        'DTD_pool_rule':'all sorted official train1, 1880 images','no_real_anomaly_development_set':True}
    config.update({'probe_source_sha256':{name:digest(ROOT/name) for name in ['localization_data.py','localization_synthesis.py','localization_train.py']},
        'probe_results_sha256':digest(OUT/'probe_results.json'),
        'synthesis_check_sha256':digest(OUT/'synthesis_check.json'),
        'DTD_train1_sha256':digest(ROOT/'data/dtd/labels/train1.txt'),
        'diagnostic_pairs_visually_reviewed':32,
        'resource_warning':'CPU free RAM fell below0.5GiB during probe; keep other apps closed. Bank peak is estimated, not measured.',
        'timing_scope':'CPU generation, host transfer, forward, loss, finite-gradient check, backward and Adam update; CUDA synchronized'})
    if (OUT/'training_config.json').exists():
        old=json.loads((OUT/'training_config.json').read_text(encoding='utf-8'))
        if old != config:
            raise ValueError('Frozen training config differs; do not overwrite')
    else:
        save(OUT/'training_config.json',config)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('day2_complete'):
        if digest(OUT/'training_config.json') != protocol['training_config_sha256']:
            raise ValueError('Frozen training config identity differs')
        print(json.dumps({'day2_already_frozen':True,'counts':split['counts'],'steps':steps,'lambda_weak':weak}),flush=True)
        return
    protocol.update({'phase':'day2_protocol_frozen','day2_complete':True,'day3_ready':True,
        'training_config':config,'training_config_sha256':digest(OUT/'training_config.json'),
        'split_sha256':digest(SPLIT),'state':'training-side probe complete; formal models not trained'})
    protocol['scientific_state'].update({'split_frozen':True,'N':len(split['N']),'Vnew':len(split['Vnew']),
        'Tfinal':len(split['Tfinal']),'lambda_weak_frozen':True,'training_budget_frozen':True,
        'model_instantiated':True,'model_training':'resource probe only, no retained weights',
        'new_memory_bank':False,'Vnew_or_Tfinal_scoring':False})
    protocol['remaining']=['Day3 new PatchCore bank and formal DRAEM-A training','Day4 paired B/repeats and Vnew calibration','Day5 frozen real test evaluation and figures']
    save(OUT/'protocol.json',protocol)
    print(json.dumps({'counts':split['counts'],'training':config},ensure_ascii=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','probe','fit-patchcore','train','calibrate','evaluate','predict'])
    parser.add_argument('--condition',choices=['A','B'],default='A')
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--stage',choices=['all','score','metrics'],default='all')
    parser.add_argument('--image')
    parser.add_argument('--model',choices=['PatchCore','A20261020','B20261020'],default='B20261020')
    parser.add_argument('--output-dir')
    args=parser.parse_args()
    if args.command=='prepare':
        print(json.dumps(preparation()['counts']),flush=True)
    elif args.command=='probe':
        probe()
    elif args.command=='fit-patchcore':
        from localization_evaluation import fit_patchcore
        fit_patchcore()
    elif args.command=='calibrate':
        from localization_calibration import calibrate
        calibrate()
    elif args.command=='evaluate':
        from localization_testing import evaluate
        evaluate(args.stage)
    elif args.command=='predict':
        if not args.image:parser.error('predict requires --image')
        from localization_testing import predict_single
        predict_single(args.image,args.model,args.output_dir)
    else:
        from localization_training import train
        train(args.condition,args.seed)

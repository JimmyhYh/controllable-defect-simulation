"""Independent admission for six manual formal runs; unchanged trainer reused."""
from contextlib import contextmanager
from pathlib import Path
import sys

from . import common as c
from . import training as t
from . import metrics as old_metrics
from . import probe_retry as retry
from . import probe_retry_metrics as pm
from .sampling import verify_pairs

CONTROL=c.RESULTS/'formal_admission_preparation_v1'
MODELS=c.MODELS/'formal_gate_v1'
RESULTS=c.RESULTS/'formal_gate_v1'
ORDER=[(condition,seed) for seed in c.SEEDS for condition in ('A','B')]
START_RAM=4.5


def cpu_guard(split):
    c.guard(split,cpu=False)
    c.require(not any(n.split('.')[0] in ('torch','torchvision','diffusers','transformers') for n in sys.modules),'Fresh CPU-only process required')
    def audit(event,args):
        if event=='import' and args[0].split('.')[0] in ('torch','torchvision','diffusers','transformers'):
            raise RuntimeError('No model imports in formal preparation/verification')
        if event in ('subprocess.Popen','pickle.find_class'):
            raise RuntimeError('No subprocess/pickle in formal preparation/verification')
    sys.addaudithook(audit)


def gate_evidence():
    """Pure hash/JSON verification: no persistent CPU-only model import hook.

    Never call pm.verify_results/input_identity or retry.verify_retry here:
    their standalone CPU guards intentionally prevent subsequent model imports.
    This verifier must also run before the human-started trainer initializes.
    """
    split,source=c.source_metadata()
    retry.verify_preparation(cpu=False); retry.verify_binding()
    with retry.retry_context():
        checked=t.verify_run('probe',retry.SEED,split,source)
    meta=checked['result']
    c.require(meta['initial_hash']==c.read_json(retry.CONTROL/'protocol.json')['expected_initial_parameter_sha256'],'Probe initialization differs')
    prep=c.verify_manifest(pm.CONTROL)
    cfg=c.read_json(pm.CONTROL/'protocol.json')
    identity=cfg['identity']
    items=c.read_json(c.PREP/'probe.json')
    required=[c.PREP/'complete.json',c.PREP/'common_config.json',c.PREP/'probe.json',retry.CONTROL/'complete.json',
              retry.CONTROL/'protocol.json',retry.NEW_RESULTS/'binding.json',pm.RUN/'complete.json',
              pm.RUN/'result.json',pm.WEIGHTS/'run.json']
    required += [pm.WEIGHTS/f'checkpoint_{step:05d}.{suffix}' for step in (200,400,600,800,1000) for suffix in ('pt','json')]
    required += [path for i,item in enumerate(items) for path in (c.PREP/item['file'],pm.RUN/'probabilities'/f'{i:02d}.npy')]
    expected_files={path.relative_to(c.ROOT).as_posix():c.sha(path) for path in required}
    expected={'files':expected_files,'frozen_core':c.identities(),'retry_adapter_sha256':retry.source_hash(),
              'metric_adapter_sha256':c.sha(pm.__file__),'environment':c.environment(),
              'initial_parameter_hash_recorded':meta['initial_hash'],'final_parameter_hash_recorded':meta['final_hash'],
              'final_checkpoint_sha256':c.sha(pm.WEIGHTS/'checkpoint_01000.pt')}
    c.require(identity==expected and cfg==pm.config(expected) and prep['identity']==expected
              and set(prep['files'])=={'protocol.json','input_validation.json'},'New gate preparation/input identity differs')
    outer=c.verify_manifest(pm.OUTPUT); inner=c.verify_manifest(pm.OUTPUT/'exact')
    names={'transaction.json','metrics.json','exact/transaction.json','exact/counts.uint32',
           'exact/positive.uint32','exact/metrics.json','exact/complete.json'}
    c.require(set(outer['files'])==names and outer['identity']==expected and inner['identity']==expected
              and outer['preparation_sha256']==c.sha(pm.CONTROL/'complete.json'),'Metric result identity/namespace differs')
    summary=c.read_json(pm.OUTPUT/'metrics.json'); exact=c.read_json(pm.OUTPUT/'exact'/'metrics.json')
    pm.validate_summary(summary,exact,meta,expected)
    c.require(summary['gate_passed'] is True,'Final technical fitting gate failed; no formal admission')
    return split,source,{'metric_complete_sha256':c.sha(pm.OUTPUT/'complete.json'),
                        'metric_preparation_sha256':c.sha(pm.CONTROL/'complete.json'),
                        'metric_json_sha256':c.sha(pm.OUTPUT/'metrics.json'),
                        'probe_complete_sha256':c.sha(pm.RUN/'complete.json'),
                        'retry_protocol_sha256':c.sha(retry.CONTROL/'protocol.json'),
                        'retry_binding_sha256':c.sha(retry.NEW_RESULTS/'binding.json'),
                        'retry_source_sha256':retry.source_hash(),'metric_source_sha256':c.sha(pm.__file__),
                        'input_identity':source,'environment':c.environment(),'core_implementation':c.identities(),
                        'AP':summary['AP'],'gate_passed':True}


def protocol(evidence):
    return {'schema':'fewshot-formal-admission-v1','gate_evidence':evidence,
            'adapter_sha256':c.sha(__file__),'common_config_sha256':c.sha(c.PREP/'common_config.json'),
            'preparation_complete_sha256':c.sha(c.PREP/'complete.json'),
            'sequence_sha256':{str(seed):c.sha(c.PREP/str(seed)/'sequence.json') for seed in c.SEEDS},
            'synthetic_manifest_sha256':c.sha(c.PREP/'synthetic_manifest.json'),
            'order':[[cond,seed] for cond,seed in ORDER],'steps':10000,'synthetic_inputs':7500,
            'model':'same frozen author discriminative3/2 base16 with GN8; random seed initialization',
            'loss':'unchanged valid CE+Dice','probe_weights_used':False,'FP32':True,'batch':1,'workers':0,
            'start_RAM_GiB':START_RAM,'runtime_RAM_stop_GiB':3,'free_VRAM_GiB':6,'free_disk_GiB':80,
            'LR_boundaries':{str(i):c.lr_at(i) for i in (1,8000,8001,9000,9001,10000)},
            'checkpoint_steps':list(range(1000,10001,1000)),
            'models':str(MODELS),'results':str(RESULTS),'automatic_training':False,
            'restore':'only own complete saved points with RNG/optimizer/cursor/LR; GPU restore not exercised',
            'formal_conditions':'A real5000+normal5000; B real5000+normal2500+synthetic2500; per pair same initialization and sequence'}


def prepare():
    split,source=c.source_metadata(); cpu_guard(split)
    _,_,evidence=gate_evidence()
    if (CONTROL/'complete.json').exists(): return verify_admission(evidence)
    c.require(not CONTROL.exists() and not MODELS.exists() and not RESULTS.exists(),'Unknown/partial formal destinations; preserve and investigate')
    resources=c.resource(3,80)
    paired=verify_pairs(split,source)
    CONTROL.mkdir()
    c.save(CONTROL/'protocol.json',protocol(evidence))
    c.save(CONTROL/'CPU_validation.json',{'resources':resources,'sequences':3,'effective_synthetic':paired['effective_synthetic'],
                                        'order':[[cond,seed] for cond,seed in ORDER],
                                        'model_instances':0,'Torch_imported':False,'AP_recomputed':False,'training_started':False})
    c.seal(CONTROL,{'protocol_sha256':c.sha(CONTROL/'protocol.json'),'adapter_sha256':c.sha(__file__)})
    return verify_admission(evidence)


def verify_admission(evidence=None):
    if evidence is None: _,_,evidence=gate_evidence()
    complete=c.verify_manifest(CONTROL)
    c.require(set(complete['files'])=={'protocol.json','CPU_validation.json'}
              and complete['protocol_sha256']==c.sha(CONTROL/'protocol.json')
              and complete['adapter_sha256']==c.sha(__file__)
              and c.read_json(CONTROL/'protocol.json')==protocol(evidence),'Formal admission protocol/implementation differs')
    return {'formal_admission_verified':True,'AP':evidence['AP'],'gate_passed':True,
            'formal_order':[[cond,seed] for cond,seed in ORDER],'steps_each':10000,
            'start_RAM_GiB':START_RAM,'runtime_RAM_stop_GiB':3,'new_model_operations':0}


def binding():
    return {'protocol_sha256':c.sha(CONTROL/'protocol.json'),'adapter_sha256':c.sha(__file__),
            'models':str(MODELS),'results':str(RESULTS),'order':[[cond,seed] for cond,seed in ORDER]}


def validate_roots(allow_binding_only=False):
    labels={f'{cond}_{seed}' for cond,seed in ORDER}
    if RESULTS.exists():
        c.require(c.read_json(RESULTS/'binding.json')==binding(),'Formal run binding differs')
        c.require({p.name for p in RESULTS.iterdir()} <= labels|{'binding.json'},'Unknown formal result entries')
    if MODELS.exists():
        c.require(RESULTS.exists() and {p.name for p in MODELS.iterdir()}<=labels,'Unknown formal model entries')


def routed_folders(condition,seed):
    c.require((condition,seed) in ORDER,'Only fixed formal A/B seeds allowed')
    return MODELS/f'{condition}_{seed}',RESULTS/f'{condition}_{seed}'


@contextmanager
def formal_context():
    folders,gate,resource=t.folders,old_metrics.verify_probe_metrics,c.resource
    def admitted_gate(split=None,identity=None):
        _,source,evidence=gate_evidence()
        if identity is not None: c.require(identity==source,'Trainer input role identity differs')
        return verify_admission(evidence)
    def headroom(min_ram=2,min_disk=80):
        return resource(max(min_ram,START_RAM),min_disk)
    t.folders=routed_folders; old_metrics.verify_probe_metrics=admitted_gate; c.resource=headroom
    try: yield
    finally:
        t.folders=folders; old_metrics.verify_probe_metrics=gate; c.resource=resource


def train(condition,seed):
    c.require((condition,seed) in ORDER,'Invalid formal condition/seed')
    # Do not install CPU-only import hooks in this explicitly manual entry.
    verify_admission()
    if not RESULTS.exists():
        c.require(not MODELS.exists(),'Orphan formal model root; preserve and investigate')
        RESULTS.mkdir()
        c.save(RESULTS/'binding.json',binding())
    validate_roots()
    with formal_context():
        return t.train(condition,seed)


def verify_run(condition,seed):
    split,source=c.source_metadata(); cpu_guard(split)
    verify_admission(); validate_roots()
    c.require(RESULTS.exists(),'Formal runs not started')
    folders=t.folders
    t.folders=routed_folders
    try:
        result=t.verify_run(condition,seed,split,source)
        meta=result['result']
        c.require(meta['exposures']=={'real':5000,'common_normal':2500,'intervention':2500},'Formal exposure totals differ')
        c.require(meta['synthetic_intervention']==(2500 if condition=='B' else 0)
                  and meta['normal_intervention']==(2500 if condition=='A' else 0),'Intervention totals differ')
        if condition=='B':
            other=t.verify_run('A',seed,split,source)
            c.require(other['result']['initial_hash']==meta['initial_hash'],'Paired initialization differs')
        return result
    finally: t.folders=folders


def status(condition,seed):
    split,_=c.source_metadata(); cpu_guard(split)
    verify_admission(); validate_roots()
    model,result=routed_folders(condition,seed)
    if (result/'complete.json').exists(): return verify_run(condition,seed)
    if result.exists(): return {'status':'partial formal result; stop and investigate'}
    if not model.exists(): return {'status':'not started; waiting for manual command','condition':condition,'seed':seed}
    if (model/'active.json').exists() or (model/'failure.json').exists():
        return {'status':'active/stale/failed; preserve and diagnose before resume'}
    return {'status':'own saved points only; manual resume subject to full identity',
            'checkpoints':t.checkpoint_inventory(model,1000,10000)}

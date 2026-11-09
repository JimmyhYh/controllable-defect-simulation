"""Independent RAM-headroom retry, reusing the unchanged frozen trainer.

Only train_retry() may call the training loop. Preparation/status/verification
do not instantiate models or optimizers. Original failed directories stay put.
"""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys

from . import common as c
from . import training as t
from . import model as m
from .sampling import verify_pairs

CONTROL = c.RESULTS / 'probe_retry_preparation_v1'
NEW_MODELS = c.MODELS / 'retry_ram_v1'
NEW_RESULTS = c.RESULTS / 'retry_ram_v1'
START_RAM = 4.5
SEED = c.PROBE_SEED


def source_hash():
    return c.sha(__file__)


def pid_is_active(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        c.require(ctypes.get_last_error()==87,'Old PID liveness unknown; preserve and investigate')
        return False
    try:
        code=wintypes.DWORD()
        c.require(kernel.GetExitCodeProcess(handle,ctypes.byref(code)), 'Old PID status query failed')
        return code.value==259
    finally:
        kernel.CloseHandle(handle)


def old_failure():
    folder=c.MODELS/f'probe_{SEED}'
    c.require({p.name for p in folder.iterdir()}=={'run.json','active.json','failure.json'},'Original failed namespace differs')
    run=c.read_json(folder/'run.json'); failure=c.read_json(folder/'failure.json'); active=c.read_json(folder/'active.json')
    c.require(failure['token']==active['token'] and failure['last_completed_update']==100
              and failure['message']=='RAM sample below3GiB: stop; do not silently change resources', 'Failure evidence differs')
    c.require(not pid_is_active(active['pid']),'Old PID still active or reused; no retry')
    c.require(not (c.RESULTS/f'probe_{SEED}').exists(),'Unexpected original final result')
    return run,{n:c.sha(folder/n) for n in ('run.json','active.json','failure.json')}


def protocol():
    split,identity=c.source_metadata()
    run,old_hashes=old_failure()
    expected=run['identity']
    c.require(expected['input_identity']==identity and expected['implementation']==c.identities()
              and expected['environment']==c.environment() and expected['preparation']==c.sha(c.PREP/'complete.json')
              and expected['config']==c.sha(c.PREP/'common_config.json'), 'Original run/input implementation identity differs')
    return split,{'schema':'fewshot-probe-ram-retry-v1','seed':SEED,'steps':1000,'start_RAM_GiB':START_RAM,
                  'runtime_RAM_stop_GiB':3,'free_VRAM_GiB':6,'input_identity':identity,
                  'preparation_sha256':expected['preparation'],'common_config_sha256':expected['config'],
                  'frozen_implementation':c.identities(),'retry_module_sha256':source_hash(),
                  'environment':c.environment(),'original_failure_files':old_hashes,
                  'expected_initial_parameter_sha256':expected['initial_hash'],
                  'models':str(NEW_MODELS/f'probe_{SEED}'),'results':str(NEW_RESULTS/f'probe_{SEED}'),
                  'intervention':'Only independent output routing and higher startup RAM headroom; original runtime stop unchanged',
                  'checkpoint_steps':[200,400,600,800,1000],
                  'LR_boundaries':{str(i):c.lr_at(i,True) for i in (1,800,801,900,901,1000)},
                  'automatic_training':False,'original_weights_loaded':False}


def cpu_guard(split):
    c.guard(split,cpu=False)
    c.require(not any(n.split('.')[0] in ('torch','torchvision','diffusers','transformers') for n in sys.modules), 'CPU retry preparation must not load models')
    def audit(event,args):
        if event=='import' and args[0].split('.')[0] in ('torch','torchvision','diffusers','transformers'):
            raise RuntimeError('Model import forbidden in retry preparation/status')
        if event in ('subprocess.Popen','pickle.find_class'):
            raise RuntimeError('No subprocess or pickle in CPU retry preparation/status')
    sys.addaudithook(audit)


def prepare_retry():
    split,expected=protocol(); cpu_guard(split)
    if (CONTROL/'complete.json').exists(): return verify_preparation(cpu=False)
    c.require(not CONTROL.exists() and not NEW_MODELS.exists() and not NEW_RESULTS.exists(),
              'Unknown/incomplete retry directories; preserve and investigate')
    verify_pairs(split,expected['input_identity'])
    CONTROL.mkdir()
    c.save(CONTROL/'protocol.json',expected)
    c.seal(CONTROL,{'protocol_sha256':c.sha(CONTROL/'protocol.json'),'retry_module_sha256':source_hash()})
    return verify_preparation(cpu=False)


def verify_preparation(cpu=True):
    split,expected=protocol()
    if cpu: cpu_guard(split)
    complete=c.verify_manifest(CONTROL)
    c.require(set(complete['files'])=={'protocol.json'} and c.read_json(CONTROL/'protocol.json')==expected
              and complete['protocol_sha256']==c.sha(CONTROL/'protocol.json')
              and complete['retry_module_sha256']==source_hash(),'Retry protocol/implementation differs')
    return {'retry_preparation_verified':True,'seed':SEED,'steps':1000,
            'start_RAM_GiB':START_RAM,'runtime_RAM_stop_GiB':3,'original_failure_preserved':True,
            'training_started':(NEW_MODELS/f'probe_{SEED}'/'run.json').exists(),
            'current_available_RAM_GiB':c.d.available_ram(),'new_model_operations':0}


def expected_binding():
    return {'protocol_sha256':c.sha(CONTROL/'protocol.json'),'retry_module_sha256':source_hash(),
            'frozen_implementation':c.identities(),'seed':SEED,
            'models':str(NEW_MODELS/f'probe_{SEED}'),'results':str(NEW_RESULTS/f'probe_{SEED}')}


def verify_binding():
    expected=expected_binding()
    c.require(c.read_json(NEW_RESULTS/'binding.json')==expected,'Retry run binding differs')
    c.require({p.name for p in NEW_RESULTS.iterdir()} <= {'binding.json',f'probe_{SEED}'},'Unknown retry result-root entries')
    c.require(NEW_MODELS.exists() and {p.name for p in NEW_MODELS.iterdir()}=={f'probe_{SEED}'},'Unknown retry model-root entries')


@contextmanager
def retry_context(for_training=False):
    """Temporary process-local adapter; frozen files remain byte-for-byte intact."""
    folders,resource,make=t.folders,c.resource,m.make_model
    def routed(condition,seed):
        c.require(condition=='probe' and seed==SEED,'This adapter only permits the fixed technical probe')
        return NEW_MODELS/f'probe_{seed}',NEW_RESULTS/f'probe_{seed}'
    def headroom(min_ram=2,min_disk=80):
        return resource(max(min_ram,START_RAM),min_disk)
    def checked_initializer(seed):
        model=make(seed)
        expected=c.read_json(CONTROL/'protocol.json')['expected_initial_parameter_sha256']
        c.require(m.state_hash(model)==expected,'Retry initial parameters differ from original initialization')
        return model
    t.folders=routed
    if for_training:
        c.resource=headroom; m.make_model=checked_initializer
    try:
        yield
    finally:
        t.folders=folders; c.resource=resource; m.make_model=make


def train_retry():
    # This is the only updating entry. Assistant preparation never calls it.
    verify_preparation(cpu=False)
    if not NEW_RESULTS.exists():
        c.require(not NEW_MODELS.exists(),'Unknown retry model root; preserve and investigate')
        NEW_RESULTS.mkdir()
        c.save(NEW_RESULTS/'binding.json',expected_binding())
    else:
        c.require(c.read_json(NEW_RESULTS/'binding.json')==expected_binding(),'Existing retry binding differs')
        c.require({p.name for p in NEW_RESULTS.iterdir()} <= {'binding.json',f'probe_{SEED}'},'Unknown retry result-root entries')
        if NEW_MODELS.exists():
            c.require({p.name for p in NEW_MODELS.iterdir()} <= {f'probe_{SEED}'},'Unknown retry model-root entries')
    with retry_context(for_training=True):
        return t.train('probe',SEED)


def verify_retry():
    verify_preparation()
    verify_binding()
    with retry_context():
        result=t.verify_run('probe',SEED)
    expected=c.read_json(CONTROL/'protocol.json')['expected_initial_parameter_sha256']
    c.require(result['result']['initial_hash']==expected,'Retry initial parameter identity differs')
    return result


def retry_status():
    prepared=verify_preparation()
    if not NEW_MODELS.exists() and not NEW_RESULTS.exists():
        return {**prepared,'status':'prepared; waiting for manual train-probe-retry'}
    if NEW_MODELS.exists(): verify_binding()
    else:
        c.require({p.name for p in NEW_RESULTS.iterdir()}=={'binding.json'}
                  and c.read_json(NEW_RESULTS/'binding.json')==expected_binding(),'Unknown pre-start retry binding')
    with retry_context():
        return {**prepared,'run_status':t.status('probe',SEED)}

"""Stage-1 CPU environment only. No image decoding entry implemented yet."""
import ctypes
import importlib
import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def inspect_environment() -> dict:
    packages={}
    for dist,module in [('numpy','numpy'),('Pillow','PIL'),('scipy','scipy')]:
        try:
            importlib.import_module(module)
            packages[dist]={'version':importlib.metadata.version(dist),'import_ok':True}
        except Exception as e:
            packages[dist]={'import_ok':False,'error':str(e)}
    try:
        import tkinter
        interp=tkinter.Tcl()
        tk={'import_ok':True,'tcl_version':str(interp.call('info','patchlevel')),
            'TkVersion':str(tkinter.TkVersion),'window_created':False,'GUI_runtime_verified':False}
    except Exception as e:
        tk={'import_ok':False,'error':str(e),'window_created':False}
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    dll=ctypes.WinDLL('kernel32',use_last_error=True)
    query=dll.GlobalMemoryStatusEx;query.argtypes=[ctypes.POINTER(Memory)];query.restype=ctypes.c_int
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not query(ctypes.byref(memory)): raise OSError(ctypes.get_last_error(),'RAM query failed')
    forbidden=[n for n in sys.modules if n.split('.')[0] in ('torch','torchvision','timm','diffusers','transformers')]
    if forbidden: raise RuntimeError('Unexpected model imports: '+str(forbidden))
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'packages':packages,'tk':tk,'RAM_total_bytes':memory.total,'RAM_available_bytes':memory.available,
            'disk_free_bytes':shutil.disk_usage(ROOT).free,'forbidden_model_imports':forbidden}

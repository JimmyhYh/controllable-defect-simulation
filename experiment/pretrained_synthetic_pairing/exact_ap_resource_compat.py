"""Isolated Win32 resource lookup for the unchanged frozen exact-AP code.

Only the resource dependency is adapted through a private globals dictionary.
No frozen module globals, DLL function prototypes or scientific inputs change.
"""
from __future__ import annotations

import ctypes
import ast
import hashlib
from pathlib import Path
import shutil
import types

from . import common as c

EXACT_SOURCE_SHA = 'a65281962dcce26e68c99234cd11c28d47c1713184486e05244c486825168948'


class _MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ('total', 'available', 'total_page', 'available_page', 'total_virtual',
         'available_virtual', 'available_extended')]


_PRIVATE_DLL = ctypes.WinDLL('kernel32', use_last_error=True)
_PRIVATE_QUERY = _PRIVATE_DLL.GlobalMemoryStatusEx
_PRIVATE_QUERY.argtypes = [ctypes.POINTER(_MemoryStatus)]
_PRIVATE_QUERY.restype = ctypes.c_int


def available_ram():
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not _PRIVATE_QUERY(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.available / 2**30


def resource(min_ram=2, min_disk=80):
    ram = available_ram()
    disk_d = shutil.disk_usage(c.ROOT).free / 2**30
    disk_e = shutil.disk_usage(c.LARGE.anchor).free / 2**30
    c.require(ram >= min_ram, f'Only {ram:.3f}GiB RAM; need {min_ram}; preserve transaction')
    c.require(disk_d >= min_disk and disk_e >= min_disk,
              f'Need {min_disk}GiB free on both D and E; preserve transaction')
    return {'available_RAM_GiB': ram, 'free_disk_GiB': disk_d,
            'E_free_disk_GiB': disk_e, 'query': 'private WinDLL function, isolated argtypes'}


class _ResourceView:
    def __init__(self, original):
        self._original = original

    resource = staticmethod(resource)

    def __getattr__(self, name):
        return getattr(self._original, name)


def _bind(original):
    original_c = original.__globals__['c']
    namespace = dict(original.__globals__)
    namespace['c'] = _ResourceView(original_c)
    bound = types.FunctionType(original.__code__, namespace, original.__name__,
                               original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = original.__kwdefaults__
    c.require(bound.__code__ is original.__code__
              and bound.__defaults__ == original.__defaults__
              and original.__globals__['c'] is original_c
              and all(bound.__globals__[key] is value for key, value in original.__globals__.items()
                      if key != 'c'), 'Exact-AP resource binding changed algorithm or original globals')
    return bound


def bind_exact_ap(frozen):
    source = Path(frozen.__file__)
    c.require(hashlib.sha256(source.read_bytes()).hexdigest() == EXACT_SOURCE_SHA,
              'Frozen exact-AP source changed')
    c.require(frozen.BINS == 0x3F800001 and frozen.CHUNK == 1048576,
              'Frozen exact-AP constants changed')
    return _bind(frozen.exact_ap)


def probe():
    """CPU resource/scalar reproduction only; no arrays, metrics or models."""
    from fewshot_synthetic import data as legacy
    c.resource_snapshot()
    try:
        legacy.available_ram()
    except ctypes.ArgumentError as error:
        reproduced = str(error)
    else:
        raise RuntimeError('Expected shared-prototype conflict did not reproduce; stop and diagnose')
    original_prototype = ctypes.windll.kernel32.GlobalMemoryStatusEx.argtypes
    snapshot = resource(3, 80)
    c.require(ctypes.windll.kernel32.GlobalMemoryStatusEx.argtypes is original_prototype,
              'Private query changed shared WinDLL prototype')
    c.require(_PRIVATE_QUERY is not ctypes.windll.kernel32.GlobalMemoryStatusEx,
              'Private resource function unexpectedly shares the global function object')

    # A fictitious function exercises the same dependency binding, never exact_ap.
    sentinel = types.SimpleNamespace(resource=lambda *_: 'original-sentinel', constant=17)
    namespace = {'c': sentinel}
    exec(compile('def scalar():\n    return c.resource(3, 80), c.constant\n', '<fictitious-resource-binding>', 'exec'), namespace)
    original = namespace['scalar']
    bound = _bind(original)
    fixed, constant = bound()
    c.require(original() == ('original-sentinel', 17) and constant == 17,
              'Fictitious dependency binding changed original function')
    # Compile only the hash-verified function definition. Never execute its body.
    from fewshot_synthetic import common as legacy_common
    source = c.ROOT / 'fewshot_synthetic/metrics.py'
    c.require(hashlib.sha256(source.read_bytes()).hexdigest() == EXACT_SOURCE_SHA,
              'Frozen exact-AP source changed during static binding probe')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'exact_ap')
    namespace = {'c':legacy_common}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    original_exact = namespace['exact_ap']
    frozen_static = types.SimpleNamespace(__file__=str(source), BINS=0x3F800001,
                                         CHUNK=1048576, exact_ap=original_exact)
    bound_exact = bind_exact_ap(frozen_static)
    bound_exact.__globals__['c'].resource(3, 80)
    c.require(original_exact.__globals__['c'] is legacy_common
              and bound_exact.__code__ is original_exact.__code__
              and bound_exact.__globals__['c'].require is legacy_common.require
              and bound_exact.__globals__['c'].seal is legacy_common.seal,
              'Static exact-AP binding altered frozen algorithm services')
    return {
        'shared_prototype_failure_reproduced': True,
        'original_error': reproduced,
        'private_query_passed': True,
        'shared_prototype_unmodified_by_adapter': True,
        'same_code_object_private_globals_binding_passed': True,
        'frozen_exact_function_definition_bound_without_calling_body': True,
        'resources': snapshot,
        'fictitious_bound_resources': fixed,
        'exact_AP_called': False,
        'GT_probability_reads': 0,
        'Torch_model_operations': 0,
    }

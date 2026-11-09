"""Read-only source resolution and explicit CPU six-model reference assembly."""
import argparse
import json

from . import normal_calibration_continuation as x

c = x.c


def references(context):
    identity, requests, _, _ = context
    refs = []
    for i, row in enumerate(identity['models']):
        if i < 4:
            ref = x.verify_cache(x.old.OUTPUT / row['label'], row, requests, x.old.binding(identity), payload=True)
        else:
            ref = x.verify_model_internal(row['label'], context, payload=True)
        refs.append(ref)
    c.require(len(refs) == 6 and [r['label'] for r in refs] == [r['label'] for r in identity['models']],
              'Need all six original ordered model sources')
    return refs


def sources_value(context, refs):
    return {'schema': 'normal-calibration-six-source-references-v1', 'binding': x.binding(),
            'Q_requests_digest': c.digest_json(context[1]), 'models': refs,
            'threshold_storage': 'original source summary.json only; no copied/recomputed thresholds',
            'original_failure_preserved': True}


def verification_value():
    return {'source_models': 6, 'normal_images_per_model': 200, 'unique_normal_images': 200,
            'full_image_evaluations': 1200, 'window_forwards_recorded': 5064,
            'same_pool_not_independent_validation': True, 'thresholds_recomputed': 0,
            'new_model_operations': 0, 'Q_E_U_original_images_or_GT_read': 0}


def verify_internal(context=None, refs=None):
    context = context if context is not None else x.prepared_context()
    refs = refs if refs is not None else references(context)
    sealed = c.verify_manifest(x.ASSEMBLY)
    c.require(set(sealed['files']) == {'sources.json', 'verification.json'} and sealed['binding'] == x.binding()
              and c.read_json(x.ASSEMBLY / 'sources.json') == sources_value(context, refs)
              and c.read_json(x.ASSEMBLY / 'verification.json') == verification_value(),
              'Reference assembly identity/source differs')
    return refs


def assemble():
    x.install_guard(writes=(x.ASSEMBLY,))
    c.resource(2, x.old.MIN_DISK)
    context = x.prepared_context()
    refs = references(context)
    if (x.ASSEMBLY / 'complete.json').exists():
        verify_internal(context, refs)
    else:
        c.require(not x.ASSEMBLY.exists(), 'Partial/unknown assembly; preserve and diagnose')
        x.ASSEMBLY.mkdir()
        c.save(x.ASSEMBLY / 'sources.json', sources_value(context, refs))
        c.save(x.ASSEMBLY / 'verification.json', verification_value())
        c.seal(x.ASSEMBLY, {'binding': x.binding()})
        verify_internal(context, refs)
    return {'assembled': True, **verification_value()}


def verify_assembly():
    x.install_guard()
    verify_internal()
    return {'assembly_verified': True, **verification_value()}


def resolve_calibrations():
    """Pure read-only resolver. No permanent CPU import hook: future E caller owns its boundary.

    This function never imports Torch, scores an input, writes a file, or alters a threshold.
    CLI prepare/assembly are separately guarded. Returned paths are validated source references.
    """
    refs = verify_internal()
    return [{'label': ref['label'], 'checkpoint': str(c.ROOT / ref['checkpoint']),
             'checkpoint_sha256': ref['checkpoint_sha256'], 'source_complete_sha256': ref['complete_sha256'],
             'probabilities': str(c.ROOT / ref['folder'] / 'probabilities'),
             'scores': str(c.ROOT / ref['scores']), 'scores_sha256': ref['files']['scores.json'],
             'summary': str(c.ROOT / ref['summary']), 'summary_sha256': ref['files']['summary.json']}
            for ref in refs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('assemble', 'verify'))
    args = parser.parse_args()
    print(json.dumps(assemble() if args.command == 'assemble' else verify_assembly(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

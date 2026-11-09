"""Future CPU metric-stage cache references. No GT decoder or metrics implemented."""
import argparse
import json
from . import real_evaluation as r


def resolve_evaluation_caches():
    """Pure read-only six-source resolver; future GT caller owns its read boundary.

    No permanent CPU hook here, as a future metric caller must authorize GT reads
    separately. No labels, original images, threshold quantiles, AP, or models are
    read/run by this resolver. A partial set never becomes a successful subset.
    """
    context = r.prepared_context()
    present = r.runtime_prefix(context)
    r.c.require(present == list(r.ORDER), 'Need all six complete E1004 caches; no successful intersection')
    refs = [r.verify_cached_model(label, context, payload=True) for label in r.ORDER]
    return {'schema': 'real-E1004-six-cache-references-v1', 'binding': r.binding(),
            'models': refs, 'images_per_model': r.IMAGES, 'unique_E_images': r.IMAGES,
            'native_pixels_per_model': r.PIXELS, 'window_forwards_per_model': r.WINDOWS,
            'original_Q_thresholds_reference_only': True, 'metrics_computed': False,
            'GT_or_original_images_read': False, 'new_model_operations': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify-all',))
    parser.parse_args()
    r.install_guard()
    result = resolve_evaluation_caches()
    print(json.dumps({'all_six_verified': True, 'models': [x['label'] for x in result['models']],
                      'images_per_model': r.IMAGES, 'native_pixels_per_model': r.PIXELS,
                      'metrics_computed': False, 'new_model_operations': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

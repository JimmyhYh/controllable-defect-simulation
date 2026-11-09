"""Complete E cache verification and future CPU metric input parser; no GT read."""
from __future__ import annotations
from . import real_scoring as r


def checked_probability(root, request, row):
    import numpy as np
    name = r.probability_path(request)
    r.c.require(row['probability'] == name and row['index'] == request['index'] and row['id'] == request['id'] and row['source_image_sha256'] == request['image_sha256'] and row['window_forwards'] == request['window_count'], 'E row order/source differs')
    r.c.require(r.t.sha(root / name) == row['probability_sha256'], 'E saved probability identity differs')
    prob = np.load(root / name, allow_pickle=False)
    r.c.require(prob.dtype == np.float32 and list(prob.shape) == request['native_shape'] and np.isfinite(prob).all() and 0 <= prob.min() <= prob.max() <= 1, 'Invalid E native probability')
    r.c.require(type(row['score']) is float and row['score'] == float(prob.max()), 'E native maximum differs')
    return prob


def verify_results(group, seed, p, req, models):
    r.t.resources(); root, _ = r.paths(group, seed); scores = r.verify_saved_metadata(group, seed, p, req, models)
    for request, row in zip(req, scores['items']):
        prob = checked_probability(root, request, row); del prob
    return {'E_scores_verified': True, 'group': group, 'seed': seed, 'images': 1004, 'windows': 4270, 'native_pixels': 146411528,
            'original_threshold': scores['original_threshold'], 'additional_forwards': 0, 'GT_reads_or_metrics': 0, 'threshold_recalculation': False}


def iter_metric_inputs(group, seed, p, req, models, metric_sources):
    """Future authorized CPU caller installs cache guard + verifies preparation.

    Verifies complete cache then yields GT metadata, probability and score in
    original order. Does not open GT or compute metrics. Not called in preparation.
    """
    verify_results(group, seed, p, req, models); root, _ = r.paths(group, seed)
    r.c.require(metric_sources == r.t.load(r.PREP / 'metric_sources.json') and req == r.t.load(r.PREP / 'requests.json'), 'Metric caller metadata differs from frozen E/GT binding')
    r.c.require(len(metric_sources) == len(req) == 1004, 'Metric metadata count differs')
    rows = r.t.load(root / 'scores.json')['items']
    for request, source, row in zip(req, metric_sources, rows):
        r.c.require(source['index'] == request['index'] and source['id'] == request['id'] and source['native_shape'] == request['native_shape'], 'GT metadata alignment differs')
        yield source, checked_probability(root, request, row), row['score']

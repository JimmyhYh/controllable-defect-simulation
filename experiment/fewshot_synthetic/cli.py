"""Explicit CPU-only commands; no training entry exists in day1."""
import argparse
import json
from . import data
from .verification import verify_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare-data', 'verify-data', 'prepare-pairs', 'verify-pairs',
                                          'train-probe', 'train-formal', 'verify-probe', 'verify-formal',
                                          'training-status', 'probe-metrics', 'prepare-probe-retry',
                                          'verify-probe-retry', 'probe-retry-status', 'train-probe-retry',
                                          'prepare-probe-retry-metrics','verify-probe-retry-metrics-preparation',
                                          'compute-probe-retry-metrics','verify-probe-retry-metrics',
                                          'prepare-formal-admission','verify-formal-admission',
                                          'train-admitted-formal','verify-admitted-formal','admitted-formal-status'))
    parser.add_argument('--condition', choices=('probe','A','B'),default='probe')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.command in ('prepare-data','verify-data'):
        if args.command == 'prepare-data': data.prepare_data()
        result = verify_data()
    elif args.command in ('prepare-pairs','verify-pairs'):
        from .sampling import prepare_pairs,verify_pairs
        result = prepare_pairs() if args.command=='prepare-pairs' else verify_pairs()
    elif args.command in ('prepare-formal-admission','verify-formal-admission','train-admitted-formal',
                          'verify-admitted-formal','admitted-formal-status'):
        from . import formal_admission as formal
        if args.command=='prepare-formal-admission': result=formal.prepare()
        elif args.command=='verify-formal-admission':
            split,_=formal.c.source_metadata(); formal.cpu_guard(split)
            result=formal.verify_admission()
        else:
            if args.condition not in ('A','B') or args.seed not in formal.c.SEEDS:
                parser.error('Explicit --condition A/B and fixed --seed required')
            commands={'train-admitted-formal':formal.train,'verify-admitted-formal':formal.verify_run,
                      'admitted-formal-status':formal.status}
            result=commands[args.command](args.condition,args.seed)
    elif args.command=='probe-metrics':
        from .metrics import probe_metrics
        result=probe_metrics()
    elif args.command in ('prepare-probe-retry-metrics','verify-probe-retry-metrics-preparation',
                          'compute-probe-retry-metrics','verify-probe-retry-metrics'):
        from . import probe_retry_metrics as metric
        if args.seed is not None and args.seed!=metric.r.SEED: parser.error('Metric seed must be2026101906')
        commands={'prepare-probe-retry-metrics':metric.prepare,
                  'verify-probe-retry-metrics-preparation':metric.verify_preparation,
                  'compute-probe-retry-metrics':metric.compute,'verify-probe-retry-metrics':metric.verify_results}
        result=commands[args.command]()
    elif args.command in ('prepare-probe-retry','verify-probe-retry','probe-retry-status','train-probe-retry'):
        from . import probe_retry as retry
        if args.seed is not None and args.seed!=retry.SEED: parser.error('Retry seed must be2026101906')
        commands={'prepare-probe-retry':retry.prepare_retry,'verify-probe-retry':retry.verify_retry,
                  'probe-retry-status':retry.retry_status,'train-probe-retry':retry.train_retry}
        result=commands[args.command]()
    else:
        from . import common as c
        from .training import train,verify_run,status
        condition='probe' if args.command in ('train-probe','verify-probe') else args.condition
        seed=args.seed if args.seed is not None else (c.PROBE_SEED if condition=='probe' else c.SEEDS[0])
        if args.command=='train-probe': result=train('probe',seed)
        elif args.command=='train-formal':
            parser.error('--condition A or B required') if condition=='probe' else None
            result=train(condition,seed)
        elif args.command in ('verify-probe','verify-formal'): result=verify_run(condition,seed)
        else: result=status(condition,seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Independent explicit E entry; prepare/verify never open image or probability."""
import argparse
import json
from . import real_scoring_common as r

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "run", "verify-results"])
    args = parser.parse_args()
    # Complete run reuse gets CPU-only admission even through the run command.
    actual_forward = args.command == "run" and not r.RUN.exists()
    r.guard(cpu=not actual_forward, raw_e=actual_forward, probabilities=args.command in {"run", "verify-results"})
    from .real_scoring import prepare, verify_preparation, preparation_summary, run
    if args.command == "prepare": result = prepare()
    elif args.command == "verify":
        p, _ = verify_preparation(); result = preparation_summary(p, True)
    elif args.command == "run": result = run()
    else:
        from .real_cache import verify_results
        result = verify_results()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()

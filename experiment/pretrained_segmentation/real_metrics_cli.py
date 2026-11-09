"""Explicit independent CPU metrics entry; prepare/verify never compute metrics."""
import argparse
import json
from . import real_metrics_common as m

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "compute", "verify-results"])
    args = parser.parse_args()
    fresh_compute = args.command == "compute" and not m.RUN.exists()
    m.guard(gt=fresh_compute, arrays=args.command in {"compute", "verify-results"}, writing=args.command in {"prepare", "compute"})
    from .real_metrics import prepare, verify_preparation, summary, compute, verify_results
    if args.command == "prepare": result = prepare()
    elif args.command == "verify":
        p, _ = verify_preparation(); result = summary(p, True)
    elif args.command == "compute": result = compute()
    else: result = verify_results()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()

"""Independent explicit normal calibration entry; prepare never runs a model."""
import argparse
import json
from . import calibration_common as a

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "run", "verify-results"])
    args = parser.parse_args()
    a.guard(cpu=args.command != "run", raw_q=args.command == "run", probability_cache=args.command in {"run", "verify-results"})
    from .normal_calibration import prepare, verify_preparation, run, verify_results
    if args.command == "prepare": result = prepare()
    elif args.command == "verify":
        p, req = verify_preparation()
        result = {"calibration_ready": True, "Q_metadata_images": len(req), "planned_windows": p["planned_window_forwards"], "raw_Q_GT_or_probabilities_read": False, "actual_threshold_computed": False, "model_imports_or_forwards": 0}
    elif args.command == "run": result = run()
    else: result = verify_results()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()

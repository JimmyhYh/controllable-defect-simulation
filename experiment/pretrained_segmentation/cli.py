"""Thin explicit entries; preparation never imports a model/training runtime."""
import argparse
import json
from . import common as c

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare-inputs", "verify-inputs", "prepare-pilot", "verify-pilot", "train-probe", "verify-probe", "score-probe", "verify-probe-scores", "probe-metrics", "verify-probe-metrics"])
    p.add_argument("--resume-from", default=None, help="manual training only: latest complete checkpoint of this own run")
    p.add_argument("--review-file", default=None, help="prepare-pilot only: actual full fixed21 visual review JSON")
    args = p.parse_args()
    c.require(args.resume_from is None or args.command == "train-probe", "Resume argument is training-only")
    c.require(args.review_file is None or args.command == "prepare-pilot", "Review argument is preparation-only")
    c.guard(cpu=args.command not in {"train-probe", "score-probe"})
    if args.command in {"prepare-inputs", "verify-inputs"}:
        from . import data
        from .preparation import strict_directories
        if c.INPUT.exists(): strict_directories(c.INPUT)
        if args.command == "prepare-inputs":
            result = data.prepare_inputs()
        else:
            r = data.verify_inputs(); result = {"cpu_inputs_ready": True, "windows": 21, "valid_pixels": r["valid_pixels"], "positive_pixels": r["positive_pixels"]}
        strict_directories(c.INPUT)
    elif args.command in {"prepare-pilot", "verify-pilot"}:
        from .preparation import prepare_pilot, verify_pilot
        if args.command == "prepare-pilot":
            c.require(c.PREP.exists() or args.review_file is not None, "First preparation requires actual saved visual review")
            result = prepare_pilot(c.load(args.review_file) if args.review_file else None)
        else:
            r, _, _ = verify_pilot(); result = {"pilot_ready_for_manual_run": True, "steps": 2000, "valid_pixels": r["valid_pixels"], "positive_pixels": r["positive_pixels"], "training_performed_by_this_command": False}
    elif args.command in {"train-probe", "verify-probe"}:
        from . import training
        result = training.train_probe(args.resume_from) if args.command == "train-probe" else training.verify_run()
    elif args.command in {"score-probe", "verify-probe-scores"}:
        from . import scoring
        result = scoring.score_probe() if args.command == "score-probe" else {"scores_verified": True, "images": len(scoring.verify_scores()["items"]), "additional_forwards": 0}
    else:
        from . import metrics
        result = metrics.compute_probe() if args.command == "probe-metrics" else metrics.verify_metrics()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

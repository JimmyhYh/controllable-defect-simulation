"""Independent formal entry; never modifies the sealed pilot CLI or globals."""
import argparse
import json
from . import common as c
from . import formal_common as f

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "train", "verify-training"])
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    c.require(args.resume_from is None or args.command == "train", "Only manual train accepts own resume point")
    f.guard(cpu=args.command != "train")
    if args.command in {"prepare", "verify"}:
        from .formal_preparation import prepare, verify
        if args.command == "prepare": result = prepare()
        else:
            p, items, seq = verify(decode=True)
            result = {"formal_ready_for_manual_training": True, "seed": p["seed"], "updates": len(seq), "training_pool_images": len(items), "training_executed": False}
    else:
        from .formal_training import train, verify_training
        result = train(args.resume_from) if args.command == "train" else verify_training()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

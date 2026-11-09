"""Explicit CPU-only entry points; no generation or model imports."""
import argparse
import json
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-inputs', 'verify-inputs'))
    args = parser.parse_args()
    from .preparation import prepare_inputs, verify_inputs
    result = prepare_inputs() if args.command == 'prepare-inputs' else verify_inputs()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

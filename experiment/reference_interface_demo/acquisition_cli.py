"""Explicit resource acquisition and read-only verification, never inference."""
import argparse
import json
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('acquire', 'verify-acquisition'))
    args = parser.parse_args()
    from .acquisition import acquire, verify_acquisition
    result = acquire() if args.command == 'acquire' else verify_acquisition()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

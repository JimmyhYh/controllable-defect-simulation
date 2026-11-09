"""Explicit CPU-only preparation commands."""
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare-inputs','verify-inputs'])
    args=parser.parse_args()
    from .preparation import prepare_inputs, verify_inputs
    result=prepare_inputs() if args.command=='prepare-inputs' else verify_inputs()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

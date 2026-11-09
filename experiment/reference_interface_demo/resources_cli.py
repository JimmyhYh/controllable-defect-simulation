"""Offline static resource preparation; no model or image entry point."""
import argparse
import json
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-resources', 'verify-resources', 'inspect-local-resources'))
    args = parser.parse_args()
    from .resources import prepare_resources, verify_resources, inspect_local_resources
    function = {'prepare-resources': prepare_resources, 'verify-resources': verify_resources,
                'inspect-local-resources': inspect_local_resources}[args.command]
    print(json.dumps(function(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

"""Independent v2 CPU preparation and explicit visual finalization."""
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare-domain')
    sub.add_parser('verify-domain')
    final=sub.add_parser('finalize-domain')
    final.add_argument('--decision',choices=['passed','failed','uncertain'],required=True)
    final.add_argument('--reason',required=True)
    args=parser.parse_args()
    if args.command=='verify-domain':
        from .domain_verification_v2 import verify_domain_inputs_v2
        result=verify_domain_inputs_v2()
    else:
        from .domain_preparation_v2 import prepare_domain_inputs_v2,finalize_visual_v2
        result=prepare_domain_inputs_v2() if args.command=='prepare-domain' else finalize_visual_v2(args.decision,args.reason)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

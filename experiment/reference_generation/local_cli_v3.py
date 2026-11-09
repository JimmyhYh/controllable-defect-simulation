"""Explicit independent CPU-only v3 preparation/review/verification."""
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare-local');sub.add_parser('verify-local')
    review=sub.add_parser('review-local')
    review.add_argument('--annotations-json',required=True,
        help='Inline JSON with structure_rectangles and uncertain_rectangles for b0,b1')
    args=parser.parse_args()
    if args.command=='prepare-local':
        from .local_preparation_v3 import prepare_local_scope_v3
        result=prepare_local_scope_v3()
    elif args.command=='verify-local':
        from .local_verification_v3 import verify_local_inputs_v3
        result=verify_local_inputs_v3()
    else:
        from .local_review_v3 import evaluate_local_review_v3
        data=json.loads(args.annotations_json)
        if set(data)!={'structure_rectangles','uncertain_rectangles'}:raise ValueError('Unexpected annotation schema')
        result=evaluate_local_review_v3(data['structure_rectangles'],data['uncertain_rectangles'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

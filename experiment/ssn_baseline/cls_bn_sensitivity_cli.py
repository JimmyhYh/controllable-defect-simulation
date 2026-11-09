"""分类卷积BN统计来源敏感性诊断：薄CLI。

只做参数分派，不含计算实现。`run` 是唯一会导入Torch并执行前向的命令；
其余命令均为纯CPU只读核验。使用边界见 README_cls_bn_sensitivity.md。
"""
from __future__ import annotations

import argparse
import json

from .cls_bn_sensitivity import (prepare, run_diagnostic, summarize,
                                 verify_diagnostic, verify_preparation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["prepare", "verify-preparation", "run", "verify", "summarize"])
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare()
    elif args.command == "verify-preparation":
        result = verify_preparation()
    elif args.command == "run":
        result = run_diagnostic()
    elif args.command == "verify":
        result = verify_diagnostic()
    else:
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

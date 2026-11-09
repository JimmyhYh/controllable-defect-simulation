"""阶段 1 命令行入口。

所有运行步骤由用户手动启动；本模块不做任何串联调用。
"""

from __future__ import annotations

import argparse
import json

from . import protocol as P


def cmd_prepare(_args) -> None:
    P.write_json(P.out_dir("preparation") / "protocol.json", {
        "version": P.VERSION,
        "field_of_view_mm": P.FIELD_OF_VIEW_MM,
        "input_size": P.INPUT_SIZE,
        "k_mm_per_px": P.K_MM_PER_PX,
        "pixel_area_mm2": P.PIXEL_AREA_MM2,
        "bank_images": P.BANK_IMAGES,
        "calibration_quantile": P.CALIB_QUANTILE,
        "semu_levels": list(P.SEMU_LEVELS),
        "polarity": P.POLARITY,
        "shapes": P.SHAPES,
        "amplitude_limit": P.AMPLITUDE_LIMIT,
        "amp_tolerance": P.AMP_TOLERANCE,
        "amp_max_iter": P.AMP_MAX_ITER,
        "backgrounds": P.background_paths(),
        "bank_source": P.bank_paths(),
        "calibration_source": P.calib_paths(),
        "scope": "method pre-check on MVTec leather; not a display result; no training",
    })
    print("protocol written:", P.out_dir("preparation") / "protocol.json")


def cmd_build_bank(args) -> None:
    from . import bank256
    print(json.dumps(bank256.run(device=args.device, overwrite=args.overwrite),
                     ensure_ascii=False, indent=1))


def cmd_render(args) -> None:
    from . import stimuli
    index = stimuli.build(overwrite=args.overwrite)
    print("stimuli:", index["count_stimuli"], "baseline:", index["count_baseline"],
          "problems:", len(index["problems"]))


def cmd_measure(args) -> None:
    from . import measure
    print(json.dumps(measure.run(device=args.device, overwrite=args.overwrite),
                     ensure_ascii=False, indent=1))


def cmd_report(_args) -> None:
    from . import report
    s = report.run()
    print(json.dumps({k: s[k] for k in ("baseline_false_positive", "baseline_unstable",
                                       "monotonic", "crosses_20_to_80", "usable",
                                       "saturated", "flat_zero", "insufficient",
                                       "criterion_passed")},
                     ensure_ascii=False, indent=1))


def cmd_real_semu(_args) -> None:
    from . import real_semu
    s = real_semu.run()
    print(json.dumps({"n_images": s["n_images"], "problems": len(s["problems"]),
                      "guard_px": s["guard_px"], "total_SEMU": s["total"]["SEMU"],
                      "by_kind_SEMU_median": {k: v["SEMU"]["median"]
                                              for k, v in s["by_kind"].items()}},
                     ensure_ascii=False, indent=1))


def cmd_s2_prepare(_args) -> None:
    from . import stage2
    print(json.dumps(stage2.prepare(), ensure_ascii=False, indent=1))


def cmd_s2_render(args) -> None:
    from . import stage2
    print(json.dumps(stage2.render_all(overwrite=args.overwrite), ensure_ascii=False, indent=1))


def cmd_s2_measure(args) -> None:
    from . import stage2
    print(json.dumps(stage2.measure_all(device=args.device, overwrite=args.overwrite),
                     ensure_ascii=False, indent=1))


def cmd_s2_report(_args) -> None:
    from . import stage2
    s = stage2.report_all()
    print(json.dumps({"per_shape": s["per_shape"], "verdict": s["verdict"],
                      "real_mura_median": s["real_mura_SEMU"]["median"],
                      "baseline_fp": s["baseline_fp"]}, ensure_ascii=False, indent=1))


def cmd_res_build(args) -> None:
    from . import resolution_probe as rp
    print(json.dumps(rp.build(args.size, overwrite=args.overwrite),
                     ensure_ascii=False, indent=1))


def cmd_res_calibrate(args) -> None:
    from . import resolution_probe as rp
    print(json.dumps(rp.calibrate(args.size, overwrite=args.overwrite),
                     ensure_ascii=False, indent=1))


def cmd_res_score(args) -> None:
    from . import resolution_probe as rp
    print(json.dumps(rp.score(args.size, overwrite=args.overwrite),
                     ensure_ascii=False, indent=1))


def cmd_res_analyze(_args) -> None:
    from . import resolution_probe as rp
    r = rp.analyze()
    print(json.dumps({"comparison": r.get("comparison"),
                      "reproduction_check": r.get("reproduction_check"),
                      "verdict": r.get("verdict")}, ensure_ascii=False, indent=1))


def cmd_verify(args) -> None:
    from . import bank256, measure, stimuli
    missing = []
    for stage, name in (("bank", "complete.json"), ("calibration", "threshold.json"),
                        ("stimuli", "complete.json"), ("scores", "scores.json")):
        if not (P.out_dir(stage) / name).exists():
            missing.append(f"{stage}/{name}")
    if missing:
        print("尚未封存，无法核验：\n  " + "\n  ".join(missing))
        print("请先按顺序完成：build-bank -> render-stimuli -> measure")
        raise SystemExit(3)

    result = {"bank": bank256.verify(device=args.device),
              "stimuli": stimuli.verify(),
              "scores": measure.verify()}
    result["passed"] = all(v["passed"] for v in result.values() if isinstance(v, dict))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if not result["passed"]:
        raise SystemExit(2)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pod_probe", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="write frozen protocol (no model, no image read)")

    b = sub.add_parser("build-bank", help="build 256x256 PatchCore bank and calibrate tau")
    b.add_argument("--device", default="cuda")
    b.add_argument("--overwrite", action="store_true")

    r = sub.add_parser("render-stimuli", help="render baseline + 7 SEMU levels x 2 shapes")
    r.add_argument("--overwrite", action="store_true")

    m = sub.add_parser("measure", help="single pass inference over sealed stimuli")
    m.add_argument("--device", default="cuda")
    m.add_argument("--overwrite", action="store_true")

    sub.add_parser("report", help="aggregate, apply criteria, draw curve")

    sub.add_parser("real-semu", help="stage 2a: SEMU of real AnoLCD defects (CPU only)")

    sub.add_parser("stage2-prepare", help="2b: fix backgrounds and protocol")
    r2 = sub.add_parser("stage2-render", help="2b: render 4 shapes x 8 levels x 60 backgrounds")
    r2.add_argument("--overwrite", action="store_true")
    m2 = sub.add_parser("stage2-measure", help="2b: forward the frozen AnoLCD PatchCore bank")
    m2.add_argument("--device", default="cuda")
    m2.add_argument("--overwrite", action="store_true")
    sub.add_parser("stage2-report", help="2b: aggregate, interpolate SEMU_50/90, verdict")

    for name, helptext in (("res-build", "stage 3: build bank at a given input size"),
                           ("res-calibrate", "stage 3: calibrate tau on Vnew"),
                           ("res-score", "stage 3: score Tfinal"),
                           ("res-analyze", "stage 3: compare sizes and apply criteria")):
        p3 = sub.add_parser(name, help=helptext)
        if name != "res-analyze":
            p3.add_argument("--size", type=int, required=True, choices=[256, 512])
            p3.add_argument("--overwrite", action="store_true")

    v = sub.add_parser("verify-all", help="read-only verification of bank, stimuli, scores")
    v.add_argument("--device", default="cuda")

    args = ap.parse_args(argv)
    {"prepare": cmd_prepare, "build-bank": cmd_build_bank, "render-stimuli": cmd_render,
     "measure": cmd_measure, "report": cmd_report, "real-semu": cmd_real_semu,
     "stage2-prepare": cmd_s2_prepare, "stage2-render": cmd_s2_render,
     "stage2-measure": cmd_s2_measure, "stage2-report": cmd_s2_report,
     "res-build": cmd_res_build, "res-calibrate": cmd_res_calibrate,
     "res-score": cmd_res_score, "res-analyze": cmd_res_analyze,
     "verify-all": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Spike S1: helical screw constraint with contacts and a scripted mocap grasp.

Runs randomized screw-in / screw-out trials on one_leg plus negative cases, and reports success
rates and real-time factor. Exit code 0 when the M1 exit criteria hold.

    .venv/Scripts/python.exe spikes/s1_helical_screw.py --trials 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsim.sim.screw_trials import TrialParams, run_trial  # noqa: E402
from fsim.spec import get_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task = get_task("one_leg")
    rng = np.random.default_rng(args.seed)

    ok, rtfs, errs = 0, [], []
    for i in range(args.trials):
        p = TrialParams(hole=int(rng.integers(4)), twist0_deg=float(rng.uniform(-180, 180)),
                        lateral_mm=float(rng.uniform(0, 1.5)), lateral_dir_deg=float(rng.uniform(0, 360)),
                        tilt_deg=float(rng.uniform(0, 3)), rate_deg_s=float(rng.uniform(60, 180)),
                        press_mm=float(rng.uniform(0.5, 2.0)), screw_timeout_s=16.0)
        r = run_trial(task, p)
        good = r.engaged and r.seated and r.assembled and r.released_at_start and r.lifted_out
        ok += good
        rtfs.append(r.rtf)
        if r.engaged:
            errs.append(r.seat_pos_err_mm)
        if not good:
            print(f"  FAIL trial {i}: {p}\n     -> engaged={r.engaged} seated={r.seated} assembled={r.assembled} "
                  f"released={r.released_at_start} lifted={r.lifted_out}")
    rate = ok / args.trials
    print(f"randomized screw-in/out: {ok}/{args.trials} = {100 * rate:.1f}%  "
          f"seat error max {max(errs):.3f} mm  RTF min {min(rtfs):.1f}x median {np.median(rtfs):.1f}x")

    negatives = {
        "stud on slab surface, 20 mm off the hole": TrialParams(lateral_mm=20.0, unscrew=False, screw_timeout_s=6),
        "tilted 15 deg": TrialParams(tilt_deg=15.0, unscrew=False, screw_timeout_s=6),
        "twisting the wrong way": TrialParams(tighten=False, unscrew=False, screw_timeout_s=8),
    }
    neg_ok = True
    for name, p in negatives.items():
        r = run_trial(task, p)
        print(f"negative [{name}]: engaged={r.engaged}")
        neg_ok &= not r.engaged
    passed = rate >= 0.95 and neg_ok and min(rtfs) > 1.0
    print("S1", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

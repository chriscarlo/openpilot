#!/usr/bin/env python3
"""
Scan qlogs for lead-following longitudinal incidents and rank segments.

Designed to answer: "I know the car misbehaved while following leads, but I
don't remember when." Point it at a realdata root (on-device
/data/media/0/realdata or a local pull) and it scores every segment for
lead-follow anomaly signatures, so you only pull full rlogs for the segments
worth study.

Signatures scored per segment (all require openpilot engaged + lead present
unless noted):
  hard_brake      sustained aEgo below threshold
  near_collision  TTC = dRel / closing-speed under threshold at speed
  gap_collapse    time-headway collapse below threshold at speed
  overbrake       strong ego decel while the lead itself is not decelerating
                  and the gap is not critical (braking harder than situation)
  oscillation     repeated large accel-command sign flips within a window
                  (hunting / surge-brake cycling)
  takeover_brake  driver pressed brake while engaged with a lead (panic exit)
  fcw             forward collision warning / AEB events (any engagement state)
  undershoot      lead present, large gap, ego well below set speed, yet
                  near-zero accel command sustained (hanging back)

Usage (on device):
  cd /data/openpilot && /usr/local/venv/bin/python3 -u \
      /tmp/scan_lead_incidents.py --root /data/media/0/realdata \
      --json-out /tmp/lead_incidents.json
Usage (dev box, on pulled segments):
  .venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/scan_lead_incidents.py \
      --root ./pulled_realdata --json-out ./lead_incidents.json

Only qlog decimated rates are assumed (carState ~10Hz, radarState ~4Hz,
longitudinalPlan ~2Hz, onroadEvents ~1Hz).
"""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path


def _find_segments(root: Path) -> list[Path]:
    segs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        for name in ("qlog.zst", "qlog.bz2", "qlog"):
            if (entry / name).exists():
                segs.append(entry / name)
                break
    return segs


def _interp_state(samples: list[tuple[float, float]], t: float, max_age: float = 1.5):
    """Latest sample at or before t, if fresh enough. samples must be time-sorted."""
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None
    ts, val = samples[lo - 1]
    return val if (t - ts) <= max_age else None


class SegmentScan:
    def __init__(self, seg_path: Path):
        self.seg_path = seg_path
        self.events: list[dict] = []
        self.counts: dict[str, int] = defaultdict(int)
        self.score = 0.0
        self.mono_start = None
        self.engaged_time_s = 0.0
        self.lead_time_s = 0.0
        self.error = None

    def add(self, kind: str, t: float, weight: float, detail: str):
        rel_t = (t - self.mono_start) if self.mono_start else 0.0
        self.events.append({"kind": kind, "t_rel_s": round(rel_t, 1), "detail": detail})
        self.counts[kind] += 1
        self.score += weight

    def summary(self) -> dict:
        return {
            "segment": str(self.seg_path.parent.name),
            "path": str(self.seg_path),
            "score": round(self.score, 1),
            "engaged_time_s": round(self.engaged_time_s, 1),
            "lead_time_s": round(self.lead_time_s, 1),
            "counts": dict(self.counts),
            "events": self.events[:80],
            "error": self.error,
        }


def scan_segment(seg_path: Path, LogReader) -> SegmentScan:
    scan = SegmentScan(seg_path)

    # time-sorted state streams
    car_states = []      # (t, dict)
    leads = []           # (t, dict) radarState.leadOne
    enabled_s = []       # (t, bool)
    accel_cmds = []      # (t, float) carControl.actuators.accel
    plan_ats = []        # (t, float) longitudinalPlan.aTarget
    cruise_speeds = []   # (t, float) m/s
    fcw_events = []      # (t, names)

    try:
        for msg in LogReader(str(seg_path)):
            t = msg.logMonoTime * 1e-9
            if scan.mono_start is None:
                scan.mono_start = t
            which = msg.which()
            try:
                if which == "carState":
                    cs = msg.carState
                    car_states.append((t, {
                        "vEgo": cs.vEgo, "aEgo": cs.aEgo,
                        "brakePressed": cs.brakePressed,
                        "gasPressed": cs.gasPressed,
                    }))
                    spd = cs.cruiseState.speed
                    if spd > 0.1:
                        cruise_speeds.append((t, spd))
                elif which == "radarState":
                    l1 = msg.radarState.leadOne
                    leads.append((t, {
                        "status": bool(l1.status), "dRel": l1.dRel,
                        "vRel": l1.vRel, "vLead": l1.vLead,
                        "aLeadK": l1.aLeadK, "modelProb": l1.modelProb,
                    }))
                elif which == "selfdriveState":
                    enabled_s.append((t, bool(msg.selfdriveState.enabled)))
                elif which == "carControl":
                    if msg.carControl.enabled:
                        accel_cmds.append((t, msg.carControl.actuators.accel))
                elif which == "longitudinalPlan":
                    plan_ats.append((t, msg.longitudinalPlan.aTarget))
                elif which == "onroadEvents":
                    names = [str(e.name) for e in msg.onroadEvents]
                    hits = [n for n in names if "fcw" in n.lower() or "aeb" in n.lower() or "stockFcw" in n]
                    if hits:
                        fcw_events.append((t, hits))
            except Exception:
                continue
    except Exception:
        scan.error = traceback.format_exc(limit=1).strip().splitlines()[-1]
        return scan

    if not car_states:
        scan.error = "no carState in qlog"
        return scan

    def lead_at(t):
        ld = _interp_state(leads, t)
        return ld if (ld and ld["status"]) else None

    def engaged_at(t):
        en = _interp_state(enabled_s, t)
        return bool(en)

    # exposure times (carState ~10Hz -> 0.1s per sample)
    for t, _cs in car_states:
        if engaged_at(t):
            scan.engaged_time_s += 0.1
            if lead_at(t):
                scan.lead_time_s += 0.1

    # --- hard_brake / overbrake / near_collision / gap_collapse / takeover ---
    hard_brake_run = 0.0
    prev_brake = False
    cooldowns: dict[str, float] = defaultdict(lambda: -1e9)

    def cooled(kind: str, t: float, gap: float = 5.0) -> bool:
        if t - cooldowns[kind] >= gap:
            cooldowns[kind] = t
            return True
        return False

    for t, cs in car_states:
        eng = engaged_at(t)
        ld = lead_at(t)

        # takeover_brake: rising edge of brakePressed while engaged w/ lead
        if cs["brakePressed"] and not prev_brake and eng and ld and cooled("takeover_brake", t):
            scan.add("takeover_brake", t, 8.0,
                     f"driver brake at vEgo={cs['vEgo']:.1f} dRel={ld['dRel']:.1f}")
        prev_brake = cs["brakePressed"]

        if not (eng and ld):
            hard_brake_run = 0.0
            continue

        v, a = cs["vEgo"], cs["aEgo"]
        dRel, vRel, aLeadK = ld["dRel"], ld["vRel"], ld["aLeadK"]

        # hard_brake: sustained
        if a < -2.5:
            hard_brake_run += 0.1
            if hard_brake_run >= 0.5 and cooled("hard_brake", t):
                scan.add("hard_brake", t, 4.0,
                         f"aEgo={a:.2f} for {hard_brake_run:.1f}s vEgo={v:.1f} dRel={dRel:.1f}")
        else:
            hard_brake_run = 0.0

        # near_collision: closing fast, low TTC
        if vRel < -0.5 and 0.5 < dRel < 40.0:
            ttc = dRel / -vRel
            if ttc < 2.5 and v > 3.0 and cooled("near_collision", t):
                w = 10.0 if ttc < 1.5 else 6.0
                scan.add("near_collision", t, w,
                         f"TTC={ttc:.1f}s dRel={dRel:.1f} vRel={vRel:.2f} vEgo={v:.1f}")

        # gap_collapse: headway below 0.6s at speed
        if v > 5.0 and dRel > 0.5:
            thw = dRel / v
            if thw < 0.6 and cooled("gap_collapse", t):
                scan.add("gap_collapse", t, 5.0,
                         f"THW={thw:.2f}s dRel={dRel:.1f} vEgo={v:.1f} vRel={vRel:.2f}")

        # overbrake: ego braking hard, lead not decelerating, gap not critical
        if a < -2.0 and aLeadK > -0.5 and v > 3.0 and dRel / max(v, 0.1) > 1.2:
            if cooled("overbrake", t):
                scan.add("overbrake", t, 6.0,
                         f"aEgo={a:.2f} aLeadK={aLeadK:.2f} THW={dRel / max(v, 0.1):.2f}s vRel={vRel:.2f}")

    # --- oscillation: accel-command sign flips with amplitude, 10s windows ---
    if accel_cmds:
        win = []
        for t, acmd in accel_cmds:
            if not (engaged_at(t) and lead_at(t)):
                win = []
                continue
            win.append((t, acmd))
            while win and t - win[0][0] > 10.0:
                win.pop(0)
            flips = 0
            for i in range(1, len(win)):
                a0, a1 = win[i - 1][1], win[i][1]
                if (a0 < -0.15 and a1 > 0.15) or (a0 > 0.15 and a1 < -0.15):
                    flips += 1
            if flips >= 3:
                amp = max(x for _, x in win) - min(x for _, x in win)
                if amp > 1.5 and cooled("oscillation", t, gap=10.0):
                    scan.add("oscillation", t, 5.0,
                             f"{flips} accel sign flips in 10s, amplitude {amp:.2f} m/s^2")
                win = []

    # --- undershoot: hanging back from set speed behind distant lead ---
    under_run = 0.0
    for t, cs in car_states:
        ld = lead_at(t)
        setp = _interp_state(cruise_speeds, t, max_age=3.0)
        acmd = _interp_state(accel_cmds, t, max_age=1.0)
        ok = (engaged_at(t) and ld and setp and acmd is not None
              and cs["vEgo"] > 3.0
              and ld["dRel"] / max(cs["vEgo"], 0.1) > 2.5
              and (setp - cs["vEgo"]) > 3.0
              and acmd < 0.15
              and not cs["gasPressed"])
        if ok:
            under_run += 0.1
            if under_run >= 4.0 and cooled("undershoot", t, gap=15.0):
                scan.add("undershoot", t, 3.0,
                         f"THW={ld['dRel'] / max(cs['vEgo'], 0.1):.1f}s "
                         f"below set by {setp - cs['vEgo']:.1f} m/s, accel_cmd={acmd:.2f}")
                under_run = 0.0
        else:
            under_run = 0.0

    # --- fcw ---
    for t, names in fcw_events:
        scan.add("fcw", t, 9.0, ",".join(names))

    return scan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="realdata root containing <route--seg>/ dirs")
    ap.add_argument("--json-out", default=None, help="write full JSON report here")
    ap.add_argument("--min-score", type=float, default=0.1, help="hide segments scoring below this")
    ap.add_argument("--limit", type=int, default=0, help="scan at most N segments (0 = all)")
    args = ap.parse_args()

    try:
        from openpilot.tools.lib.logreader import LogReader
    except ImportError:
        sys.path.insert(0, os.getcwd())
        try:
            from openpilot.tools.lib.logreader import LogReader
        except ImportError:
            print("ERROR: cannot import openpilot.tools.lib.logreader; run from an openpilot checkout root", file=sys.stderr)
            return 2

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    segs = _find_segments(root)
    if args.limit:
        segs = segs[: args.limit]
    print(f"scanning {len(segs)} segments under {root} ...", flush=True)

    results = []
    for i, seg in enumerate(segs):
        scan = scan_segment(seg, LogReader)
        results.append(scan)
        flag = f" score={scan.score:.0f} {dict(scan.counts)}" if scan.score > 0 else ""
        err = f" ERROR:{scan.error}" if scan.error else ""
        print(f"[{i + 1}/{len(segs)}] {seg.parent.name}{flag}{err}", flush=True)

    ranked = sorted((r for r in results if r.score >= args.min_score), key=lambda r: -r.score)

    print("\n=== RANKED LEAD-FOLLOW INCIDENT SEGMENTS ===")
    if not ranked:
        print("(none above min-score)")
    for r in ranked[:30]:
        print(f"\n{r.summary()['segment']}  score={r.score:.0f}  "
              f"engaged={r.engaged_time_s:.0f}s lead={r.lead_time_s:.0f}s")
        for ev in r.events[:10]:
            print(f"  t+{ev['t_rel_s']:>6.1f}s  {ev['kind']:<15} {ev['detail']}")
        if len(r.events) > 10:
            print(f"  ... {len(r.events) - 10} more events")

    if args.json_out:
        report = {"root": str(root), "n_segments": len(segs),
                  "ranked": [r.summary() for r in ranked],
                  "errors": [r.summary() for r in results if r.error]}
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"\nfull report -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import os, sys, time, json, glob, argparse, threading, queue

"""
Real-time VTSC watcher: tails VTSC snapshots and swaglogs, summarizes caps and gating.

Sources:
- Snapshots: /data/media/0/VTSCDebug/vtsc_snapshots.jsonl (if enabled)
- Swaglogs:  /data/log/swaglog.* lines containing "VTSCDBG " JSON payload

Outputs concise lines with key fields and flags to detect:
- freeway_failopen_missed: physics/visibility indicate fail-open, but occlusion cap active
- double_occl_cap_suspect: raw target ~= occl cap while occl cap also active in active_cap
- pretrigger_with_high_conf: pretrigger reason while conf is good
- psi_below_thresh: occlusion active but psi_vis < psi_thresh
- map_low_coverage: map cap winning with sparse coverage (<30%) and cap significantly below v_base

Usage:
  python tools/vtsc/vtsc_watch.py [--snapshots] [--swaglog]
"""

SNAPSHOT_PATH = "/data/media/0/VTSCDebug/vtsc_snapshots.jsonl"
SWAGLOG_GLOB = "/data/log/swaglog.*"

def tail_f(path, out_q):
  try:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
      f.seek(0, os.SEEK_END)
      while True:
        line = f.readline()
        if not line:
          time.sleep(0.1)
          continue
        out_q.put(('snap', line))
  except Exception as e:
    out_q.put(('err', f"tail_f error for {path}: {e}"))

def tail_swaglogs(out_q):
  # naive multi-file tail: reopen newest periodically
  last_sizes = {}
  while True:
    try:
      files = sorted(glob.glob(SWAGLOG_GLOB))
      for p in files[-5:]:
        try:
          sz = os.path.getsize(p)
        except Exception:
          continue
        pos = last_sizes.get(p, sz)
        # on first see, jump near end to avoid backlog
        if p not in last_sizes:
          pos = max(0, sz - 65536)
        if pos < sz:
          try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
              f.seek(pos)
              for line in f:
                if ('VTSCDBG ' in line) or ('LEADROLEDBG ' in line):
                  out_q.put(('swag', line))
          except Exception:
            pass
          last_sizes[p] = sz
      time.sleep(0.2)
    except Exception as e:
      out_q.put(('err', f"tail_swaglogs error: {e}"))
      time.sleep(0.5)

def parse_snapshot_line(s):
  try:
    d = json.loads(s)
    return d if isinstance(d, dict) else None
  except Exception:
    return None

def parse_swag_line(s):
  try:
    rec = json.loads(s)
    msg = rec.get('msg')
    if not isinstance(msg, str):
      msg = rec.get('msg$s')
    if isinstance(msg, str) and msg.startswith('VTSCDBG '):
      j = json.loads(msg.split('VTSCDBG ', 1)[1])
      return ("vtscdbg", j) if isinstance(j, dict) else None
    if isinstance(msg, str) and msg.startswith('LEADROLEDBG '):
      j = json.loads(msg.split('LEADROLEDBG ', 1)[1])
      return ("leadrole", j) if isinstance(j, dict) else None
  except Exception:
    return None
  return None

def fmt_float(x, n=1):
  try:
    return f"{float(x):.{n}f}"
  except Exception:
    return "?"

def evaluate_flags(d):
  flags = []
  v = float(d.get('v', 0.0))
  cap = str(d.get('active_cap') or '')
  occluded = bool(d.get('occluded', False))
  kappa_vis = abs(float(d.get('kappa_vis', d.get('k_vis_last', 0.0) or 0.0)))
  s_vis = float(d.get('s_visible_m', d.get('vis_horizon_s', 0.0) * v))
  conf = float(d.get('path_conf', d.get('conf', 0.0)))
  psi_vis = float(d.get('psi_vis', 0.0))
  psi_thresh = float(d.get('psi_thresh', 0.0))
  reason = str(d.get('occlusion_reason') or '')
  # Fail-open miss: straight, long visibility, good confidence, but occlusion cap active
  if (kappa_vis <= 1e-5) and (s_vis >= 120.0) and (conf >= 0.60) and (cap == 'occlusion'):
    flags.append('freeway_failopen_missed')
  # Double occlusion suspicion: raw is already near occl cap while occlusion cap wins
  raw = float(d.get('raw', d.get('vtsc_cmd', 0.0)))
  cap_occ = float(d.get('cap_occl_vmin', 0.0))
  if cap == 'occlusion' and cap_occ > 0 and abs(raw - cap_occ) <= 0.5:
    flags.append('double_occl_cap_suspect')
  # Pretrigger with high confidence: reason=pretrigger but confidence not degraded
  if reason == 'pretrigger' and conf >= 0.70:
    flags.append('pretrigger_with_high_conf')
  # Occlusion while psi below threshold: gating inconsistency
  if cap == 'occlusion' and psi_vis < psi_thresh - 0.05:
    flags.append('psi_below_thresh')
  # Map low coverage: map cap winning but coverage is sparse and cap is significantly below base
  map_cov = float(d.get('map_tail_coverage', 1.0))
  cap_map = float(d.get('cap_map_vmin', 0.0))
  v_base = float(d.get('v_base', 0.0))
  if cap == 'map' and map_cov < 0.3 and v_base > 0.0 and cap_map < (v_base - 1.0):
    flags.append('map_low_coverage')
  return flags

def render_line(d, src):
  v = fmt_float(d.get('v'))
  base = fmt_float(d.get('v_base'))
  raw = fmt_float(d.get('raw', d.get('vtsc_cmd')))
  final = fmt_float(d.get('final', d.get('vtsc_cmd')))
  v_occ = fmt_float(d.get('v_occ'))
  v_vis = fmt_float(d.get('v_vis'))
  cap = d.get('active_cap')
  cap_vis = fmt_float(d.get('cap_visible_vmin'))
  cap_occ = fmt_float(d.get('cap_occl_vmin'))
  cap_map = fmt_float(d.get('cap_map_vmin'))
  psi = fmt_float(d.get('psi_vis'))
  psi_th = fmt_float(d.get('psi_thresh'))
  conf = fmt_float(d.get('conf', d.get('path_conf')))
  reason = d.get('occlusion_reason')
  tail = fmt_float(d.get('tail_frac'))
  s_tail = fmt_float(d.get('s_tail'))
  flags = evaluate_flags(d)
  flags_s = (",".join(flags)) if flags else "-"
  return (
    f"[{src}] v={v} base={base} raw={raw} final={final} | cap={cap} vis={cap_vis} occ={cap_occ} map={cap_map} | "
    f"v_vis={v_vis} v_occ={v_occ} conf={conf} psi={psi}/{psi_th} reason={reason} tail={tail}@{s_tail} | flags={flags_s}"
  )


def render_leadrole_line(d, src):
  roles = d.get("roles", {})
  reasons = d.get("reasons", {})
  raw = d.get("raw", {})
  lead0 = raw.get("lead0", {})
  lead1 = raw.get("lead1", {})
  return (
    f"[{src}] leadrole src={d.get('source')} v={fmt_float(d.get('vEgo'))} gate={d.get('gate_active')} "
    f"dup={d.get('duplicate_pair')} drop={d.get('dropped_slot')} "
    f"r0={roles.get('lead0')}({reasons.get('lead0')}) y0={fmt_float(lead0.get('yRel'))} d0={fmt_float(lead0.get('dRel'))} "
    f"r1={roles.get('lead1')}({reasons.get('lead1')}) y1={fmt_float(lead1.get('yRel'))} d1={fmt_float(lead1.get('dRel'))}"
  )

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--snapshots', action='store_true', help='Tail VTSC snapshots file')
  ap.add_argument('--swaglog', action='store_true', help='Tail swaglogs for VTSCDBG lines')
  args = ap.parse_args()
  if not args.snapshots and not args.swaglog:
    args.snapshots = args.swaglog = True

  q = queue.Queue()
  threads = []
  if args.snapshots and os.path.exists(SNAPSHOT_PATH):
    t = threading.Thread(target=tail_f, args=(SNAPSHOT_PATH, q), daemon=True)
    t.start(); threads.append(t)
    print(f"[watch] tailing snapshots at {SNAPSHOT_PATH}")
  elif args.snapshots:
    print(f"[watch] snapshot file not found: {SNAPSHOT_PATH}")
  if args.swaglog:
    t2 = threading.Thread(target=tail_swaglogs, args=(q,), daemon=True)
    t2.start(); threads.append(t2)
    print(f"[watch] tailing swaglogs at {SWAGLOG_GLOB}")

  try:
    while True:
      try:
        src, line = q.get(timeout=1.0)
      except queue.Empty:
        continue
      if src == 'err':
        print(line, file=sys.stderr)
        continue
      d = None
      if src == 'snap':
        d = parse_snapshot_line(line)
        src_tag = 'snap'
        kind = 'snap'
      elif src == 'swag':
        parsed = parse_swag_line(line)
        if parsed:
          kind, d = parsed
        else:
          kind, d = None, None
        src_tag = f"swag:{kind}" if kind else "swag"
      if not d:
        continue
      try:
        if kind == 'leadrole':
          print(render_leadrole_line(d, src_tag))
        else:
          print(render_line(d, src_tag))
      except Exception as e:
        print(f"[watch] render error: {e}")
  except KeyboardInterrupt:
    pass

if __name__ == '__main__':
  main()

# VTSC Overslow Diagnostics — GPT‑5 Pro (Execution‑Ready Primer)

Scope: branch `chubbs‑merge` • window: 2025‑09‑05 00:43–01:43 UTC

Heads‑up: These commands are copy‑pasteable locally against the repo paths below. They extract the exact counts so you can drop them straight into a PR. Adjust header names in awk if your TSV uses slightly different keys.

---

## 0) Files/paths to use

- Primer & “Patch Under Test”: `docs/chauffeur/vtsc/cases/overslow_2025-09-05/PROMPT_GPT5_PRO.md`
- Hour overview: `docs/chauffeur/vtsc/debug/debug_2025-09-05/DIAGNOSTICS_OVERVIEW.md`
- Snapshot TSV: `docs/chauffeur/vtsc/debug/debug_2025-09-05/vtsc_snapshots.tsv`
- Case A (visible‑cap heavy): `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/`
- Case B (occlusion‑cap focus): `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/`
- Off‑road FOV summaries:
  - Case B/80: `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060843/{metrics.json,by_log.jsonl}`
  - Case A/67: `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060856/{metrics.json,by_log.jsonl}`

Assumed TSV headers include: `active_cap, overslow, psi_vis, psi_thresh, raw, final, cap_occl_vmin, cap_visible_vmin, conf (or smoothed_confidence), reason (or occlusion_reason), segment_id`.

---

## 1) Hour & cases — overslow counts from snapshot TSV

What to pull:
- total rows, overslow rows
- overslow by `active_cap` (visible vs occlusion)
- overslow by `reason` (`fov_exit`, `pretrigger`, …)
- per‑case breakdown for `…--67` and `…--80`

Command (run from repo root; adjust path as needed):

```bash
TSV="docs/chauffeur/vtsc/debug/debug_2025-09-05/vtsc_snapshots.tsv"
awk -F'\t' '
NR==1 { for (i=1;i<=NF;i++) H[$i]=i; next }
{
  total++
  overslow += ($(H["overslow"])+0)>0
  cap = $(H["active_cap"])
  reason = (H["reason"]?$(H["reason"]):$(H["occlusion_reason"]))
  seg = $(H["segment_id"])
  if (($(H["overslow"])+0)>0) {
    o_by_cap[cap]++
    o_by_reason[reason]++
    if (seg ~ /00000085--f247b281ca--67/) o_case67++
    if (seg ~ /00000085--f247b281ca--80/) o_case80++
  }
}
END {
  printf "TOTAL rows: %d\nOVERSLOW rows: %d\n", total, overslow
  printf "\nOverslow by active_cap:\n"
  for (k in o_by_cap) printf "  %s: %d\n", k, o_by_cap[k]
  printf "\nOverslow by reason:\n"
  for (k in o_by_reason) printf "  %s: %d\n", k, o_by_reason[k]
  printf "\nCase overslow:\n  ...--67: %d\n  ...--80: %d\n", o_case67+0, o_case80+0
}' "$TSV"
```

Suggested PR table columns: Scope, Overslow, Visible, Occlusion, fov_exit, pretrigger, Notes

---

## 2) Psi‑gate consistency when `active_cap=occlusion`

Definition: frames where occlusion wins and `psi_vis < psi_thresh` (or `< psi_thresh − margin`).

```bash
awk -F'\t' '
NR==1{for(i=1;i<=NF;i++)H[$i]=i;next}
$(H["active_cap"])=="occlusion"{
  psi=$(H["psi_vis"])+0; th=$(H["psi_thresh"])+0
  if (psi < th) bad++
  tot++
}
END {
  printf "Occlusion-cap frames: %d\n", tot
  printf "psi below thresh while occlusion wins: %d (%.1f%%)\n", bad+0, (tot?100.0*bad/tot:0)
}' "$TSV"
```

Quote a few flagged lines (from case reports) showing `cap=occlusion` with `psi` below thresh during `fov_exit`.

---

## 3) Double‑capping (raw ≈ occl vmin while occlusion wins)

Definition: `active_cap=occlusion` and `|raw - cap_occl_vmin| ≤ 0.30 m/s`.

```bash
EPS=0.30
awk -v EPS="$EPS" -F'\t' '
NR==1{for(i=1;i<=NF;i++)H[$i]=i;next}
$(H["active_cap"])=="occlusion"{
  tot++
  raw=$(H["raw"])+0; occ=$(H["cap_occl_vmin"])+0
  if (raw>=0 && occ>=0 && (raw-occ<=EPS) && (occ-raw<=EPS)) dbl++
}
END {
  printf "Occlusion-cap frames: %d\n", tot
  printf "double-cap suspects (|raw - occl_vmin|<=%.2f): %d (%.1f%%)\n", EPS, dbl+0, (tot?100.0*dbl/tot:0)
}' "$TSV"
```

Include 1–2 short quotes showing `raw`≈`cap_occl_vmin` when occlusion wins.

---

## 4) fov_exit crawl holds near ~2.7 m/s at low confidence

Definition: `active_cap=occlusion` and `reason=fov_exit` and `final ≤ 2.9 m/s` and `conf ≤ 0.05`.

```bash
awk -F'\t' '
NR==1{for(i=1;i<=NF;i++)H[$i]=i;next}
$(H["active_cap"])=="occlusion"{
  tot++
  r = (H["reason"]?$(H["reason"]):$(H["occlusion_reason"]))
  vfin=$(H["final"])+0
  conf= ($(H["conf"])?$(H["conf"])+0 : $(H["smoothed_confidence"])+0)
  if (r=="fov_exit" && vfin<=2.9 && conf<=0.05) crawl++
}
END {
  printf "Occlusion-cap frames: %d\n", tot
  printf "fov_exit low-conf crawl holds (final<=2.9, conf<=0.05): %d\n", crawl+0
}' "$TSV"
```

Quote 1–2 examples with `conf≈0`, `reason=fov_exit`, `final≈2.7–3.2`.

---

## 5) Cross‑check with off‑road FOV reports

If the JSONL includes per‑log tallies for psi and occlusion, use `jq` to derive quick summaries:

Per‑log occlusion+psi tally:

```bash
jq -r '
  .[] |
  {log: (.log_id // .log // "unknown"),
   occl_wins: .counts.occlusion_winner // 0,
   psi_below: .counts.psi_below_thresh_while_occl // 0,
   dbl_cap: .counts.double_occl_cap_suspect // 0,
   fov_exit_crawl: .counts.fov_exit_crawl_lowconf // 0}
' docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060843/by_log.jsonl
```

Top offenders (sort by psi_below):

```bash
jq -r '
  .[] | {log, psi_below: (.counts.psi_below_thresh_while_occl // 0)} |
  select(.psi_below>0)
' docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060843/by_log.jsonl | sort -t: -k2,2nr | head
```

---

## 6) “Patch Under Test” → expected improvements

If the patch implements psi‑gated occlusion, a double‑cap guard, and a time‑based relaxer during `fov_exit` at very low confidence (plus optional z‑axis fail‑open):

- Psi‑gate consistency: ↓ `psi_below_thresh ∧ active_cap=occlusion` (esp. in …–80).
- Double‑capping: ↓ `|raw − cap_occl_vmin| ≤ 0.30` when occlusion wins.
- fov_exit crawl holds: ↓ count and dwell of `final ≤ 2.9` with `conf ≤ 0.05`.
- No regressions on freeway fail‑open or visible‑cap behavior (monitor hour‑wide split: visible vs occlusion).

---

## 7) Minimal, safe tunables (off‑road Params)

- `VTSC.PsiThreshRad = 0.020` (with `VTSC.PsiHystRad = 0.005`)
- `VTSC.DoubleCapEpsMps = 0.30`
- `VTSC.OcclConfFloor = 0.05`
- `VTSC.FovExitRelaxS = 0.60` (seconds)
- `VTSC.OcclVminNudgeMps = 0.50` (per step, clamp to visible cap)
- Optional z‑axis: `VTSC.KappaZAxisMax = 2.0e-4 1/m`, `VTSC.ZAxisHoldS = 0.40 s`, `VTSC.ZAxisFailOpen = true`

---

## 8) Short write‑up template (drop your numbers in)

> Overslow (hour + cases). In the one‑hour window we observed {X} overslow events out of {Y} frames. By cap: visible={V}, occlusion={O}. By reason: fov_exit={F}, pretrigger={P}, ….  
> Case A (…–67) shows {A} overslow (mostly visible; fov_exit={Af} / pretrigger={Ap}).  
> Case B (…–80) shows {B} overslow with fov_exit dominant; a high fraction of frames have psi below threshold while occlusion wins and several double‑cap suspects, plus crawl holds at ~2.7 m/s under low confidence.  
> Quotes:
>  • cap=occlusion … conf=0.0 psi=0.3/0.4 … flags=psi_below_thresh  
>  • final=3.1 | cap=occlusion … flags=double_occl_cap_suspect,psi_below_thresh  
> Expected Patch effects: fewer `psi_below_thresh ∧ occl`, fewer double‑cap, and shorter/rarer `fov_exit` crawls (bounded by visible cap).  
> Params to start: PsiThreshRad=0.020, PsiHystRad=0.005, DoubleCapEpsMps=0.30, OcclConfFloor=0.05, FovExitRelaxS=0.60, OcclVminNudgeMps=0.50, (optional) KappaZAxisMax=2e‑4, ZAxisHoldS=0.40, ZAxisFailOpen=true.


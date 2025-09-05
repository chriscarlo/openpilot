# VTSC Metrics Summary — 2025-09-05 (Drive 06:12–06:30 local)

Repo commit: 921fef98cf77 (branch chubbs-merge)

Sources analyzed:
- Replay snapshots (current controller over rlogs): docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined.jsonl
- Snapshot TSV: docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined.tsv
- Analyzer summary: docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined_summary.txt
- Off-road FOV-gate aggregates (static vs tuned hysteresis): docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_145922/

Highlights (Replay Analyzer)
- Snapshots: 22,800; Vision states: FULL=110, PARTIAL=11,628, SEVERE=11,062.
- LKG floor saves: 0/10,044 occlusion points (ratio 0.00%).
- Straight freeway crawl (final << v_base with FULL & straight): 0/448.
- Highway bypass failure (occlusion depressing target ≥~65 mph): 0/7,905.
- Reacquisition nudge failure (no a_cmd ≥ 0.18 m/s² within ~0.65 s after FULL): 57/110 (51.8%).
- Jerk/comfort violations (bounds check, replay-level heuristic): 6,141/22,800 (26.9%).

Highlights (Off-road FOV-Gate Evaluator)
- Freeway occluded after (median across logs): 0.0%.
- Hidden-turn recall after (% of positive psi recalled as occluded):
  - Static gate (controller default hysteresis): ~95.4% median.
  - Tuned hysteresis (n_on=2, n_off=12): ~98.9% median.
- “Crawl after” % appears ~100% due to using `v_set` baseline; prefer base vs final for crawl assessment on curvy segments.

Interpretation
- Fail-open on straight freeway is functioning: no straight-road crawl and freeway occlusion ~0%.
- Core issue: slow accel after visibility returns to FULL (reacq nudge) — shows up as ~52% failures to reach ≥0.18 m/s² within ~0.65 s.
- Vision floor-lift rarely triggers (0 “saves”), so targets stay near v_occ during intermittent occlusion even when v_vis/base are safe.
- Hidden-turn recall remains high under tuned hysteresis, implying we can be less sticky without losing recall.

Recommendations (for GPT-5 Pro to consider)
- Add explicit reacq nudge: enforce minimum a_cmd ≥ 0.18 m/s² for ~0.65 s after FULL if final < 0.98×v_base.
- Lift occlusion cap to max(v_occ, v_lkg×0.98) with 2–3 s TTL under occlusion to avoid starvation after brief dips.
- Use tuned hysteresis (n_on≈2, n_off≈12) and optionally increase psi_margin when path_conf high and s_visible long.
- Preserve freeway fail-open invariants and comfort limits.

Appendix: quick distribution checks (replay)
- kappa_vis p50/p90/p99 ≈ 0.025/0.099/0.429 1/m.
- psi_vis p50/p90 ≈ 0.34/3.94 rad; s_visible_m p50 ≈ 21 m.
- active_cap counts: visible≈22,151; occlusion≈649; map≈0.


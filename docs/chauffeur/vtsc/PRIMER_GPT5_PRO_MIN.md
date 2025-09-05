Short Primer: VTSC Overslow Diagnostics

Please open and read the full prompt and artifacts (all are static files on GitHub, no local access required):

- Full prompt: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/PROMPT_GPT5_PRO.md
- Diagnostics overview: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/debug/debug_2025-09-05/DIAGNOSTICS_OVERVIEW.md
- Case A (visible-cap): https://github.com/chriscarlo/chauffeur/tree/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67
- Case B (occlusion focus): https://github.com/chriscarlo/chauffeur/tree/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80

What to do:
- Summarize overslow patterns (hour window and both cases) using the TSVs and flagged logs.
- Check psi‑gate consistency (`psi_vis` vs `psi_thresh`) when `active_cap=occlusion`.
- Identify any double‑capping (raw ≈ `cap_occl_vmin` while occlusion wins).
- Review fov_exit behavior under near‑zero confidence (crawl holds at ~2.7 m/s) and map any sticky recovery.

Bonus (if time):
- From the off‑road FOV gating reports, note improvements a stricter psi‑gate might bring; call out any freeway fail‑open risks.


# Lessons

- Treat Windows symlink status changes as relevant when the task involves moving between WSL and the Windows GUI; verify `core.symlinks`, link type, and Python import behavior before classifying them as unrelated worktree noise.
- When discussing lead-distance noise, state whether the value is raw model/radar `dRel` noise or post-filter `hyundai_virtual_lead_debug["filtered"]["dRel"]` movement; those have different caps and failure modes.
- Do not call the Hyundai no-radar `dRel` smoother a Kalman filter unless the code actually implements a Kalman state update; the March 30, 2026 commit `dacdf89a6` added `LeadDistanceFilter`, a prediction-corrector EMA with gates, not a formal KF.

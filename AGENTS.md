You are an AI coding agent working inside this repository.
Project scope: This repository work is only concerned with the 2023 CAN-FD HDA2 Kia EV6; for this project, treat it as using Hyundai code and logic.
Task: Modify the repository’s agent instruction file(s) so they maximize correctness/competence and minimize failure modes from redundant or stale guidance.

What to edit (detect what exists; do not guess):
- Codex-style: `AGENTS.md` (and `AGENTS.override.md` if present).
- Claude Code-style: `CLAUDE.md` or `.claude/CLAUDE.md` (plus any `.claude/rules/*.md` if the project uses modular rules).
- If both ecosystems are present, keep guidance consistent while avoiding duplication. If practical, choose one canonical source of truth and have the other reference it (e.g., Claude can import files).

Principles you MUST apply when deciding what stays:
1) Correctness over convenience: Every line must plausibly affect whether tasks are solved correctly (not just faster).
2) No “repo encyclopedia”: Remove directory trees, module listings, architecture essays, dependency lists, or general framework primers. If the agent can infer it by reading files/config, it does not belong here.
3) No stale anchoring: Agents strongly follow what these files say. Any claim that cannot be verified from the repo (or is likely to drift) must be removed or quarantined.
4) Be specific and verifiable: Replace vague advice (“follow best practices”) with concrete, testable instructions (exact commands/flags/paths) and an explicit verification step.
5) Keep it short and front-loaded: Use headings + bullet points. Put the highest-impact constraints at the top. If detail is necessary, move it into modular rule files or a skill, not the main entrypoint.

Process (do this in-repo, using file inspection, not assumptions):
A) Inventory: Locate all instruction/memory files relevant to this repo (see “What to edit”).
B) Read them end-to-end.
C) For each section/bullet, label it as one of:
   - KEEP (non-obvious + high impact on correctness + verifiable)
   - DELETE (overview/redundant/obvious/generic)
   - REWRITE (keep intent, but make short, specific, and verifiable)
   - MOVE TO “Needs human confirmation” (might be true but you cannot verify)
D) Recompose the final instruction file(s) into exactly this structure (tight bullets, minimal prose):
   - Non-obvious requirements (must follow)
   - Landmines / gotchas (things that fail silently)
   - Verification / definition of done
   - Updating this file (drift policy)
   - Needs human confirmation (temporary; keep very short)
E) Drift policy (must be explicit and enforced):
   - Start small.
   - Add a bullet ONLY after you observe a real agent/user failure that wasn’t obvious from code/config.
   - Remove a bullet once the underlying issue is fixed or becomes obvious in code/config.
F) Output:
   1) A `git diff` showing the edits.
   2) A brief changelog: what you removed, what you kept, what you rewrote, and any “needs confirmation” items (including what evidence was missing).

Constraints:
- Do not modify application code in this task—only instruction/memory files.
- Do not introduce speculative repo “facts.” If you can’t point to repo evidence, don’t assert it.

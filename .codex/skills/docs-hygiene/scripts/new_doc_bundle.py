#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Final


ALLOWED_KIND: Final[set[str]] = {"debug", "experiment", "note"}


def _slugify(value: str) -> str:
  raw = value.strip().lower()
  raw = re.sub(r"[^a-z0-9]+", "-", raw)
  raw = re.sub(r"-{2,}", "-", raw)
  raw = raw.strip("-")
  if not raw:
    raise ValueError("slug cannot be empty after normalization")
  return raw


def _find_repo_root(start: Path) -> Path:
  current = start.resolve()
  for _ in range(30):
    if (current / ".git").exists():
      return current
    if current.parent == current:
      break
    current = current.parent
  return start.resolve()


def _git_short_sha(repo_root: Path) -> str | None:
  try:
    result = subprocess.run(
      ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
      check=False,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
    )
  except FileNotFoundError:
    return None
  sha = result.stdout.strip()
  return sha if (result.returncode == 0 and sha) else None


def _mkdir(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=False)


def _write(path: Path, content: str) -> None:
  path.write_text(content, encoding="utf-8")


def _readme_template(
  *,
  area: str,
  kind: str,
  slug: str,
  created_at_utc: dt.datetime,
  env: str | None,
  git_sha: str | None,
  artifact_dir_rel: str | None,
) -> str:
  title = f"{area} — {slug} ({kind})"
  created_str = created_at_utc.isoformat().replace("+00:00", "Z")
  env_str = env or "[TODO: tici|wsl|pc]"
  sha_str = git_sha or "[TODO: git SHA]"

  artifacts_lines: list[str] = ["## Artifacts", ""]
  artifacts_lines.append("- Tracked: this README and any small sanitized outputs under this folder")
  if artifact_dir_rel is not None:
    artifacts_lines.append(f"- Untracked (raw logs/traces): `{artifact_dir_rel}`")
  else:
    artifacts_lines.append("- Untracked (raw logs/traces): [disabled]")

  return "\n".join(
    [
      f"# {title}",
      "",
      f"- Created (UTC): `{created_str}`",
      f"- Env: `{env_str}`",
      f"- Git: `{sha_str}`",
      "",
      "## Summary",
      "",
      "- [TODO: 1–3 bullets: what’s happening + why it matters]",
      "",
      "## Repro / Commands",
      "",
      "```sh",
      "# [TODO: commands run, params, scenario]",
      "```",
      "",
      "\n".join(artifacts_lines),
      "",
      "## Results",
      "",
      "- [TODO: what changed; measurable impact; screenshots/plots links]",
      "",
      "## Next Steps",
      "",
      "- [TODO: follow-ups, tuning, tests to add, routes to collect]",
      "",
      "## Notes",
      "",
      "- Avoid names like `final`, `fixed`, `test`. Use stable identifiers instead.",
      "",
    ]
  )


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create a reproducible doc bundle folder (README + optional untracked artifacts dir).",
  )
  parser.add_argument("--area", required=True, help="Subsystem area (e.g., mtsc, vtsc, mapd, ui)")
  parser.add_argument("--slug", required=True, help="Short slug describing the work (will be normalized)")
  parser.add_argument(
    "--kind",
    default="debug",
    choices=sorted(ALLOWED_KIND),
    help="Doc bundle type (affects folder layout)",
  )
  parser.add_argument(
    "--docs-base",
    default="docs/chauffeur",
    help="Docs base directory (repo-relative)",
  )
  parser.add_argument(
    "--cache-base",
    default=".cache/doc_artifacts",
    help="Artifacts cache base directory (repo-relative)",
  )
  parser.add_argument("--env", default=None, help="Environment tag (tici|wsl|pc|...)")
  parser.add_argument(
    "--no-cache-dir",
    action="store_true",
    help="Do not create an untracked artifacts directory",
  )
  args = parser.parse_args()

  area = _slugify(args.area)
  slug = _slugify(args.slug)
  kind = args.kind

  script_dir = Path(__file__).resolve().parent
  repo_root = _find_repo_root(script_dir)

  created_at_utc = dt.datetime.now(dt.timezone.utc)
  date_utc = created_at_utc.date().isoformat()
  git_sha = _git_short_sha(repo_root)

  kind_folder = {
    "debug": ("debug", f"debug_{date_utc}"),
    "experiment": ("experiments", f"experiment_{date_utc}"),
    "note": ("notes", f"note_{date_utc}"),
  }[kind]
  doc_dir = repo_root / args.docs_base / area / kind_folder[0] / kind_folder[1] / slug

  artifact_dir: Path | None = None
  if not args.no_cache_dir:
    artifact_dir = repo_root / args.cache_base / area / kind_folder[1] / slug

  try:
    _mkdir(doc_dir)
    if artifact_dir is not None:
      artifact_dir.mkdir(parents=True, exist_ok=False)
  except FileExistsError as exc:
    print(f"[ERROR] Path already exists: {exc.filename}", file=sys.stderr)
    return 2

  artifact_dir_rel: str | None = None
  if artifact_dir is not None:
    artifact_dir_rel = str(artifact_dir.relative_to(repo_root))

  readme_path = doc_dir / "README.md"
  readme = _readme_template(
    area=area,
    kind=kind,
    slug=slug,
    created_at_utc=created_at_utc,
    env=args.env,
    git_sha=git_sha,
    artifact_dir_rel=artifact_dir_rel,
  )
  _write(readme_path, readme)

  print("[OK] Created doc bundle:")
  print(f"  - {readme_path.relative_to(repo_root)}")
  if artifact_dir_rel is not None:
    print("[OK] Created untracked artifacts dir:")
    print(f"  - {artifact_dir_rel}")

  print("\nNext steps:")
  print(f"  - Fill in: {readme_path.relative_to(repo_root)}")
  if artifact_dir_rel is not None:
    print(f"  - Put raw logs/traces in: {artifact_dir_rel}")
  print("  - Keep filenames stable (date/sha/env/run), not 'final-final-2'")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


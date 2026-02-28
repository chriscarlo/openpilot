#!/usr/bin/env python3
"""
Audit and report on repo instruction/context files.

Stdlib-only. Prints a Markdown report to stdout.

Primary focus:
  - AGENTS.md as the canonical instruction source
  - CLAUDE.md and .claude/CLAUDE.md symlink status
  - Instruction bloat / drift signals (suspicious headings, missing referenced paths)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


CLAUDE_ENTRYPOINTS = (Path("CLAUDE.md"), Path(".claude/CLAUDE.md"))
CANONICAL_AGENTS = Path("AGENTS.md")
AGENTS_OVERRIDE = Path("AGENTS.override.md")

SUSPICIOUS_HEADINGS = {
    "overview",
    "architecture",
    "structure",
    "directory structure",
    "repo structure",
    "design",
    "background",
    "introduction",
    "dependencies",
    "setup",
    "installation",
}

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class FileAudit:
    relpath: str
    exists: bool
    is_symlink: bool
    symlink_target: Optional[str]
    symlink_resolves_to: Optional[str]
    in_edit_scope: bool
    lines: Optional[int]
    words: Optional[int]
    headings: tuple[str, ...]
    suspicious_headings: tuple[str, ...]
    missing_references: tuple[str, ...]


def _run_git_toplevel(start: Path) -> Optional[Path]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return Path(out) if out else None


def _find_repo_root(start: Path) -> Path:
    git_root = _run_git_toplevel(start)
    if git_root is not None:
        return git_root
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def _git_ls_files(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    raw = proc.stdout.split(b"\0")
    out: list[str] = []
    for entry in raw:
        if not entry:
            continue
        try:
            out.append(entry.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(entry.decode("utf-8", errors="replace"))
    return out


def _looks_like_markdown_instruction_file(relpath: str) -> bool:
    if relpath.endswith(("AGENTS.md", "AGENTS.override.md", "CLAUDE.md")):
        return True
    if relpath.startswith(".claude/") and relpath.endswith(".md"):
        return True
    if relpath.startswith(".claude/rules/") and relpath.endswith(".md"):
        return True
    if relpath in {".cursorrules", ".windsurfrules"}:
        return True
    if relpath.startswith(".cursor/") and relpath.endswith(".md"):
        return True
    if relpath == ".github/copilot-instructions.md":
        return True
    return False


def _in_edit_scope(relpath: str) -> bool:
    if relpath in {"AGENTS.md", "AGENTS.override.md", "CLAUDE.md", ".claude/CLAUDE.md"}:
        return True
    return relpath.startswith(".claude/rules/")


def _safe_read_text(path: Path, *, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _count_lines_words(text: str) -> tuple[int, int]:
    lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    words = len(re.findall(r"\b\w+\b", text))
    return lines, words


def _extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        heading_text = m.group(2).strip()
        headings.append(heading_text)
    return headings


def _normalize_heading(h: str) -> str:
    lowered = h.strip().lower()
    lowered = re.sub(r"[`*_~]+", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _flag_suspicious_headings(headings: Iterable[str]) -> list[str]:
    suspicious: list[str] = []
    for h in headings:
        norm = _normalize_heading(h)
        if norm in SUSPICIOUS_HEADINGS:
            suspicious.append(h)
            continue
        # Also match prefix forms: "Architecture - X", "Overview: ..."
        if any(norm.startswith(s + " ") for s in SUSPICIOUS_HEADINGS):
            suspicious.append(h)
    return suspicious


def _strip_link_fragment(target: str) -> str:
    # drop URL fragments and common "line refs" like file.py:123
    target = target.split("#", 1)[0].strip()
    m = re.match(r"^(.+):(\d+)(?::(\d+))?$", target)
    if m:
        return m.group(1)
    return target


def _extract_reference_candidates(text: str) -> set[str]:
    candidates: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        for m in LINK_RE.finditer(line):
            raw = m.group(1).strip()
            if not raw:
                continue
            candidates.add(raw)

        for m in CODE_SPAN_RE.finditer(line):
            raw = m.group(1).strip()
            if not raw:
                continue
            candidates.add(raw)

    return candidates


def _looks_like_path(token: str) -> bool:
    if any(ch in token for ch in ("<", ">", "{", "}", "*")):
        return False
    if token.startswith(("#", "http://", "https://", "mailto:", "tel:")):
        return False
    if "://" in token:
        return False
    if token.startswith(("`", "!", "[")):
        return False
    if " " in token or "\t" in token:
        return False
    if "$" in token:
        # environment/path templating - not locally verifiable
        return False
    if token.startswith(("./", "../", "/", "~")):
        return True
    if "/" in token or "\\" in token:
        return True
    if re.search(r"\.[a-zA-Z0-9]{1,6}$", token):
        return True
    return False


def _sanitize_path_token(token: str) -> str:
    token = token.strip()
    token = token.strip("()[]{}<>")
    token = token.strip(".,;:")
    token = token.strip()
    return token


def _path_exists_anywhere(
    token: str, *, root: Path, relative_to: Path
) -> tuple[bool, Optional[Path]]:
    """
    Return (exists, resolved_path_if_exists).
    Checks both file-relative and repo-root-relative resolutions for relative tokens.
    """
    token = _strip_link_fragment(token)
    token = _sanitize_path_token(token)
    if not token:
        return False, None
    candidate_paths: list[Path] = []
    if token.startswith("~"):
        candidate_paths.append(Path(os.path.expanduser(token)))
    elif token.startswith("/"):
        candidate_paths.append(Path(token))
    else:
        candidate_paths.append(relative_to / token)
        candidate_paths.append(root / token)

    for p in candidate_paths:
        try:
            if p.exists():
                return True, p
        except OSError:
            continue
    return False, None


def _audit_file(
    root: Path, relpath: str, *, canonical_agents: Optional[Path]
) -> FileAudit:
    abspath = root / relpath
    exists = abspath.exists() or abspath.is_symlink()
    is_symlink = abspath.is_symlink()
    symlink_target: Optional[str] = None
    symlink_resolves_to: Optional[str] = None
    if exists and is_symlink:
        try:
            symlink_target = os.readlink(abspath)
        except OSError:
            symlink_target = None
        try:
            symlink_resolves_to = str(abspath.resolve())
        except OSError:
            symlink_resolves_to = None

    text: Optional[str] = None
    lines: Optional[int] = None
    words: Optional[int] = None
    headings: tuple[str, ...] = ()
    suspicious_headings: tuple[str, ...] = ()
    missing_refs: tuple[str, ...] = ()

    if exists:
        text = _safe_read_text(abspath)
    if text is not None:
        lines, words = _count_lines_words(text)
        extracted_headings = _extract_headings(text)
        headings = tuple(extracted_headings)
        suspicious_headings = tuple(_flag_suspicious_headings(extracted_headings))

        # Best-effort: referenced repo paths that don't exist.
        referenced = _extract_reference_candidates(text)
        missing: set[str] = set()
        for raw in referenced:
            if not _looks_like_path(raw):
                continue
            exists_any, _ = _path_exists_anywhere(raw, root=root, relative_to=abspath.parent)
            if not exists_any:
                missing.add(_sanitize_path_token(_strip_link_fragment(raw)))
        missing_refs = tuple(sorted(missing))

    in_scope = _in_edit_scope(relpath)

    # If this is a Claude entrypoint, additionally check canonicalization target.
    if exists and relpath in {str(p) for p in CLAUDE_ENTRYPOINTS} and canonical_agents:
        if is_symlink:
            try:
                if abspath.resolve() != canonical_agents.resolve():
                    # Keep it as a missing reference? no - recommendations cover.
                    pass
            except OSError:
                pass

    return FileAudit(
        relpath=relpath,
        exists=exists,
        is_symlink=is_symlink,
        symlink_target=symlink_target,
        symlink_resolves_to=symlink_resolves_to,
        in_edit_scope=in_scope,
        lines=lines,
        words=words,
        headings=headings,
        suspicious_headings=suspicious_headings,
        missing_references=missing_refs,
    )


def _format_bool(v: bool) -> str:
    return "yes" if v else "no"


def _fmt_code(s: str) -> str:
    return f"`{s}`"


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted(set(items))


def _collect_context_files(root: Path) -> list[str]:
    tracked = _git_ls_files(root)
    candidates: set[str] = set()
    for p in tracked:
        if _looks_like_markdown_instruction_file(p):
            candidates.add(p)

    # Ensure key entrypoints are considered even if untracked.
    for p in (CANONICAL_AGENTS, AGENTS_OVERRIDE, *CLAUDE_ENTRYPOINTS):
        candidates.add(str(p))

    # .claude/rules/*.md may exist untracked.
    rules_dir = root / ".claude" / "rules"
    if rules_dir.is_dir():
        for md in rules_dir.glob("*.md"):
            try:
                candidates.add(str(md.relative_to(root)))
            except ValueError:
                pass

    return sorted(candidates)


def _symlink_status_line(
    root: Path, entrypoint: Path, *, canonical_agents: Optional[Path]
) -> str:
    p = root / entrypoint
    if not (p.exists() or p.is_symlink()):
        return f"- {_fmt_code(str(entrypoint))}: none present"
    if not p.is_symlink():
        return f"- {_fmt_code(str(entrypoint))}: exists, but is **not** a symlink (needs fix)"

    try:
        link_text = os.readlink(p)
    except OSError:
        link_text = "<unreadable>"
    try:
        resolved = p.resolve()
    except OSError:
        resolved = None

    canonical_ok = None
    if canonical_agents and resolved:
        try:
            canonical_ok = resolved == canonical_agents.resolve()
        except OSError:
            canonical_ok = None

    if canonical_ok is True:
        return (
            f"- {_fmt_code(str(entrypoint))}: symlink -> {_fmt_code(link_text)} "
            f"(resolves to {_fmt_code(str(resolved))}) (OK)"
        )
    if canonical_ok is False:
        return (
            f"- {_fmt_code(str(entrypoint))}: symlink -> {_fmt_code(link_text)} "
            f"(resolves to {_fmt_code(str(resolved))}); expected {_fmt_code(str(canonical_agents))} (needs fix)"
        )
    return (
        f"- {_fmt_code(str(entrypoint))}: symlink -> {_fmt_code(link_text)} "
        f"(resolves to {_fmt_code(str(resolved) if resolved else '<unresolvable>')})"
    )


def _recommendations(
    audits: list[FileAudit], *, root: Path, canonical_agents: Optional[Path]
) -> dict[str, list[str]]:
    """
    Heuristic, deterministic recommendations suitable for human review.
    """
    cut: list[str] = []
    rewrite: list[str] = []
    verify_or_quarantine: list[str] = []
    symlink_fixes: list[str] = []

    audits_by_path = {a.relpath: a for a in audits}

    # Symlink fixes for Claude entrypoints.
    for entry in CLAUDE_ENTRYPOINTS:
        rel = str(entry)
        a = audits_by_path.get(rel)
        if not a or not a.exists:
            continue
        if not a.is_symlink:
            symlink_fixes.append(
                f"{_fmt_code(rel)} exists but is not a symlink to {_fmt_code(str(CANONICAL_AGENTS))}"
            )
            continue
        if canonical_agents and a.symlink_resolves_to:
            try:
                if (root / rel).resolve() != canonical_agents.resolve():
                    symlink_fixes.append(
                        f"{_fmt_code(rel)} points to {_fmt_code(a.symlink_target or '<unknown>')} "
                        f"but should point to {_fmt_code(str(CANONICAL_AGENTS))}"
                    )
            except OSError:
                symlink_fixes.append(f"{_fmt_code(rel)} symlink target could not be resolved (verify manually)")

    # Content heuristics.
    for a in audits:
        if not a.exists or a.words is None:
            continue

        # Bloat signal: too many words in an instruction file.
        if a.relpath.endswith((".md", ".cursorrules", ".windsurfrules")) and a.words >= 800:
            rewrite.append(f"{_fmt_code(a.relpath)} is large ({a.words} words): rewrite shorter / delete redundancy")

        # Suspicious headings.
        if a.suspicious_headings:
            cut.append(
                f"{_fmt_code(a.relpath)} contains suspicious headings: "
                + ", ".join(_fmt_code(h) for h in a.suspicious_headings)
            )

        # Unverifiable references.
        if a.missing_references:
            verify_or_quarantine.append(
                f"{_fmt_code(a.relpath)} references missing paths: "
                + ", ".join(_fmt_code(p) for p in a.missing_references[:10])
                + (" …" if len(a.missing_references) > 10 else "")
            )

        # .claude/rules should be minimal pointers if present.
        if a.relpath.startswith(".claude/rules/") and a.words >= 80:
            rewrite.append(
                f"{_fmt_code(a.relpath)} under `.claude/rules/` is non-trivial ({a.words} words): "
                "prune duplication and convert to a minimal pointer to `AGENTS.md`"
            )

    out: dict[str, list[str]] = {
        "cut": _sorted_unique(cut),
        "rewrite": _sorted_unique(rewrite),
        "verify_or_quarantine": _sorted_unique(verify_or_quarantine),
        "symlink_fixes": _sorted_unique(symlink_fixes),
    }
    return out


def _markdown_table_row(cols: list[str]) -> str:
    return "| " + " | ".join(cols) + " |"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect via git / .git).",
    )
    args = parser.parse_args(argv)

    start = args.root if args.root is not None else Path.cwd()
    root = _find_repo_root(start).resolve()

    canonical_agents = root / CANONICAL_AGENTS
    canonical_agents_path = canonical_agents if canonical_agents.exists() else None

    now = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    relpaths = _collect_context_files(root)
    audits = [_audit_file(root, rel, canonical_agents=canonical_agents_path) for rel in relpaths]

    print("# Context File Audit Report")
    print()
    print(f"- Generated: {_fmt_code(now)}")
    print(f"- Repo root: {_fmt_code(str(root))}")
    if canonical_agents_path:
        print(f"- Canonical: {_fmt_code(str(CANONICAL_AGENTS))} (OK)")
    else:
        print(f"- Canonical: {_fmt_code(str(CANONICAL_AGENTS))} (MISSING)")
    print()

    print("## Files found")
    print()
    print(
        _markdown_table_row(
            ["File", "Exists", "Symlink", "In edit scope", "Lines", "Words", "Notes"]
        )
    )
    print(_markdown_table_row(["---", "---", "---", "---", "---", "---", "---"]))

    for a in audits:
        notes: list[str] = []
        if a.relpath == str(CANONICAL_AGENTS):
            notes.append("canonical")
        if a.relpath == str(AGENTS_OVERRIDE) and not a.exists:
            notes.append("none present")
        if a.relpath in {str(p) for p in CLAUDE_ENTRYPOINTS} and not a.exists:
            notes.append("none present")
        if a.exists and a.is_symlink and a.symlink_target:
            notes.append(f"-> {a.symlink_target}")
        row = [
            _fmt_code(a.relpath),
            _format_bool(a.exists),
            _format_bool(a.is_symlink) if a.exists else "n/a",
            _format_bool(a.in_edit_scope),
            str(a.lines) if a.lines is not None else "n/a",
            str(a.words) if a.words is not None else "n/a",
            "; ".join(notes) if notes else "",
        ]
        print(_markdown_table_row(row))

    print()
    print("## Claude entrypoints (canonicalization)")
    print()
    for entry in CLAUDE_ENTRYPOINTS:
        print(_symlink_status_line(root, entry, canonical_agents=canonical_agents_path))
    print()

    print("## Headings (suspicious headings flagged)")
    print()
    for a in audits:
        if not a.exists or not a.headings:
            continue
        print(f"### {_fmt_code(a.relpath)}")
        for h in a.headings[:40]:
            flag = " (SUSPICIOUS)" if h in a.suspicious_headings else ""
            print(f"- {_fmt_code(h)}{flag}")
        if len(a.headings) > 40:
            print(f"- … ({len(a.headings) - 40} more)")
        print()

    print("## Referenced paths that do not exist (best-effort)")
    print()
    missing_any = False
    for a in audits:
        if not a.exists or not a.missing_references:
            continue
        missing_any = True
        print(f"### {_fmt_code(a.relpath)}")
        for p in a.missing_references[:50]:
            print(f"- {_fmt_code(p)}")
        if len(a.missing_references) > 50:
            print(f"- … ({len(a.missing_references) - 50} more)")
        print()
    if not missing_any:
        print("- None detected (OK)")
        print()

    recs = _recommendations(audits, root=root, canonical_agents=canonical_agents_path)
    print("## Top recommendations")
    print()

    def _print_recs(title: str, items: list[str]) -> None:
        print(f"### {title}")
        if not items:
            print("- None (OK)")
            print()
            return
        for item in items:
            print(f"- {item}")
        print()

    _print_recs("Cut (delete)", recs["cut"])
    _print_recs("Rewrite shorter", recs["rewrite"])
    _print_recs("Verify or quarantine", recs["verify_or_quarantine"])
    _print_recs("Symlink fixes required", recs["symlink_fixes"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

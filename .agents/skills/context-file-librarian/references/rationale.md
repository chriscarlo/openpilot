# Rationale (Context File Librarian)

## Why a single canonical `AGENTS.md`

When multiple instruction entrypoints drift, agents (and humans) get conflicting constraints:

- stale commands waste time and break verification loops
- duplicated “requirements” anchor the agent on incorrect assumptions
- buried landmines fail silently because they are missed during fast reads

Keeping `AGENTS.md` minimal and verifiable reduces instruction debt and makes reviews easier.

## Why enforce symlinks for `CLAUDE.md`

Tooling sometimes reads `CLAUDE.md` as the entrypoint for instructions. By symlinking those
entrypoints to `AGENTS.md`, the repo guarantees there is one truth source and prevents copy/paste
forks.

References:

- Git symlinks (conceptual): https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-symlink
- GitHub symlink behavior: https://docs.github.com/en/repositories/working-with-files/using-files/creating-symbolic-links


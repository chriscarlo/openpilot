import os

from openpilot.common.basedir import BASEDIR

VENDORED_REF_FILE = ".vendored_ref"


def _read_vendored_ref(repo_path: str) -> str | None:
  vendored_ref_path = os.path.join(repo_path, VENDORED_REF_FILE)
  try:
    with open(vendored_ref_path) as f:
      ref = f.read().strip()
    return ref or None
  except OSError:
    return None


def get_tinygrad_ref():
  repo_path = os.path.join(BASEDIR, "tinygrad_repo")
  if vendored_ref := _read_vendored_ref(repo_path):
    return vendored_ref

  git_path = os.path.join(repo_path, ".git")
  try:
    if os.path.isdir(git_path):
      git_dir = git_path
    else:
      with open(git_path) as f:
        line = f.read().strip()
      git_dir = os.path.join(repo_path, line[8:])
    with open(os.path.join(git_dir, "HEAD")) as f:
      ref = f.read().strip()
    if ref.startswith("ref:"):
      with open(os.path.join(git_dir, ref.split(" ", 1)[1])) as f:
        return f.read().strip()
    return ref
  except Exception as e:
    print(f"Error getting tinygrad_repo ref: {e}")
    return None


def main():
  current_ref = get_tinygrad_ref()
  if current_ref:
    print(current_ref)
  else:
    print("")


if __name__ == "__main__":
  main()

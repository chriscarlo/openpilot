import requests
from pathlib import Path

import sunnypilot.models.tinygrad_ref as tinygrad_ref
from sunnypilot.models.fetcher import ModelFetcher
from openpilot.common.basedir import BASEDIR


def test_tinygrad_ref_vendored_metadata():
  vendored_ref_path = Path(BASEDIR) / "tinygrad_repo" / ".vendored_ref"
  assert vendored_ref_path.read_text().strip() == tinygrad_ref.get_tinygrad_ref()


def test_tinygrad_ref_git_fallback(tmp_path, monkeypatch):
  repo_path = tmp_path / "tinygrad_repo"
  git_dir = repo_path / ".git"
  refs_dir = git_dir / "refs" / "heads"
  refs_dir.mkdir(parents=True)
  (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
  (refs_dir / "main").write_text("deadbeef\n")
  monkeypatch.setattr(tinygrad_ref, "BASEDIR", str(tmp_path))
  assert tinygrad_ref.get_tinygrad_ref() == "deadbeef"


def fetch_tinygrad_ref():
  response = requests.get(ModelFetcher.MODEL_URL, timeout=10)
  response.raise_for_status()
  json_data = response.json()
  return json_data.get("tinygrad_ref")


def test_tinygrad_ref():
  current_ref = tinygrad_ref.get_tinygrad_ref()
  remote_ref = fetch_tinygrad_ref()
  assert remote_ref == current_ref, (
    f"""tinygrad_repo ref does not match remote tinygrad_ref of current compiled driving models json.
  Current: {current_ref}
  Remote: {remote_ref}
  Please run build-all workflow to update models."""
  )
  print("tinygrad_repo ref matches current compiled driving models json ref.")

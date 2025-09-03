#!/usr/bin/env python3
import os, json, glob, subprocess, sys, pathlib, pytest

REPO = pathlib.Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs" / "chauffeur" / "vtsc" / "offroad" / "eval_fov_gate_on_rlogs.py"

def _find_manifest():
  env = os.environ.get("VTSC_MANIFEST")
  if env and os.path.exists(env):
    return env
  cands = sorted(glob.glob(str(REPO / "docs" / "chauffeur" / "vtsc" / "vtsc_fov_fix_*" / "MANIFEST.json")))
  return cands[-1] if cands else None

@pytest.mark.offroad
def test_offroad_freeway_no_crawl_and_hidden_recall():
  if not SCRIPT.exists():
    pytest.skip(f"Eval script not found at {SCRIPT}")
  manifest = _find_manifest()
  if not manifest:
    pytest.skip("No MANIFEST.json found; set VTSC_MANIFEST or place one under vtsc_fov_fix_*/")
  outdir = REPO / "docs" / "chauffeur" / "vtsc" / "offroad" / "reports"
  cmd = [sys.executable, str(SCRIPT), "--manifest", manifest, "--outdir", str(outdir)]
  res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
  print(res.stdout)

  metrics_path = None
  for line in res.stdout.splitlines()[::-1]:
    if "Artifacts written to:" in line:
      metrics_path = line.split("Artifacts written to:")[-1].strip()
      break
  if not metrics_path:
    pytest.fail("Eval did not complete; no metrics directory found in output")

  with open(os.path.join(metrics_path, "metrics.json"), "r") as f:
    m = json.load(f)["aggregate"]

  assert (m.get("freeway_occluded_after_pct_med") or 0.0) <= 2.0, "Freeway occlusion after gating too high"
  assert (m.get("crawl_after_pct_med") or 0.0) <= 1.0, "Freeway crawl after gating too high"
  if m.get("hidden_recall_after_pct_med") is not None:
    assert m["hidden_recall_after_pct_med"] >= 90.0, "Hidden-turn recall after gating too low"

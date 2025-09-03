import os, glob, json, importlib.util, pathlib, pytest

# test file path: docs/chauffeur/vtsc/testing/integration/test_comm_starvation_diagnostics.py
BASE = pathlib.Path(__file__).resolve().parents[3] / "vtsc"

def latest_data_dir():
  env = os.environ.get("VTSC_DATA_DIR")
  if env and os.path.isdir(env): return env
  cands = sorted(glob.glob(str(BASE / "vtsc_good_data_*")))
  if not cands: pytest.skip("No vtsc_good_data_* directory found")
  return cands[-1]

def load_json(p):
  with open(p, "r") as f: return json.load(f)

def import_compiler():
  path = pathlib.Path(__file__).resolve().parents[3] / "vtsc" / "tools" / "compile_diagnostics.py"
  if not path.exists():
    pytest.skip("compile_diagnostics.py not found")
  spec = importlib.util.spec_from_file_location("compile_diagnostics", str(path))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)  # type: ignore
  return mod

def test_rlog_rates_shape():
  dd = latest_data_dir()
  rr = os.path.join(dd, "rlog_rates.json")
  if not os.path.isfile(rr):
    pytest.skip("rlog_rates.json not present")
  data = load_json(rr)
  assert isinstance(data, dict) and data, "rlog_rates.json should be a non-empty dict"
  # validate minimal shape
  for rlog, payload in data.items():
    assert "duration_s" in payload and "topics" in payload, f"Missing keys in {rlog}"
    topics = payload["topics"]
    for k in ["liveCalibration","driverMonitoringState","longitudinalPlan","liveDelay","liveParameters","radarState","liveTorqueParameters","driverAssistance","mapTurnSpeedControlSP"]:
      assert k in topics, f"Topic {k} missing in {rlog}"
      stats = topics[k]
      assert all(s in stats for s in ("n","hz_med","p95_gap_s")), f"Stats missing for {k} in {rlog}"

def test_compile_and_verdict_allowed():
  dd = latest_data_dir()
  mod = import_compiler()
  final = mod.compile_diagnostics(dd)
  for k in ["runtime","services","live_probe","load_hint","verdict","notes"]:
    assert k in final, f"Missing '{k}' in diagnostics"
  assert final["verdict"] in {"starvation","service_schema","logging_pressure","inconclusive"}
  assert isinstance(final["notes"], str)
  # If starvation, notes should carry key details
  if final["verdict"] == "starvation":
    s = final["notes"]
    assert "cls=" in s and "rtprio=" in s and "Worst p95 gap offenders" in s

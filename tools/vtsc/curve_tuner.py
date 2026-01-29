#!/usr/bin/env python3
import argparse
import ast
import datetime as _dt
import json
import math
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk


MS_TO_MPH = 2.2369362920544
MPH_TO_MS = 0.44704


def _clip(x: float, lo: float, hi: float) -> float:
  return max(lo, min(float(x), hi))


def _safe_float(x: object, default: float) -> float:
  try:
    return float(x)
  except Exception:
    return float(default)


def _now_iso() -> str:
  return _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _repo_root() -> str:
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git_head_short() -> str:
  try:
    out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_repo_root(), stderr=subprocess.DEVNULL)
    return out.decode("utf-8", errors="ignore").strip()
  except Exception:
    return "unknown"


def _read_text(path: str) -> str:
  with open(path, "r", encoding="utf-8", errors="ignore") as f:
    return f.read()


def _write_text(path: str, content: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)


def _append_jsonl(path: str, record: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")


def _load_jsonl(path: str) -> list[dict]:
  out: list[dict] = []
  try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          rec = json.loads(line)
        except Exception:
          continue
        if isinstance(rec, dict):
          out.append(rec)
  except FileNotFoundError:
    pass
  return out


def _logspace_desc(kappa_max: float, kappa_min: float, n: int) -> list[float]:
  if n <= 1:
    return [float(kappa_max)]
  kappa_max = max(float(kappa_max), 1e-12)
  kappa_min = max(float(kappa_min), 1e-12)
  log_max = math.log10(kappa_max)
  log_min = math.log10(kappa_min)
  out = []
  for i in range(n):
    t = i / float(n - 1)
    log_k = log_max + t * (log_min - log_max)
    out.append(10.0 ** log_k)
  return out


def _lerp(a: float, b: float, t: float) -> float:
  return float(a) + (float(b) - float(a)) * float(t)


def _smoothstep(t: float) -> float:
  t = _clip(t, 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def _interp_piecewise_linear(xs: list[float], ys: list[float], x: float) -> float:
  if not xs or len(xs) != len(ys):
    raise ValueError("invalid interp arrays")
  if len(xs) == 1:
    return float(ys[0])
  if x <= xs[0]:
    return float(ys[0])
  if x >= xs[-1]:
    return float(ys[-1])
  lo = 0
  hi = len(xs) - 1
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if x < xs[mid]:
      hi = mid
    else:
      lo = mid
  x0 = float(xs[lo])
  x1 = float(xs[hi])
  y0 = float(ys[lo])
  y1 = float(ys[hi])
  if x1 <= x0:
    return float(y0)
  t = (float(x) - x0) / (x1 - x0)
  return _lerp(y0, y1, t)


def _interp_logx(points: list[tuple[float, float]], kappa: float, *, default: float = 1.0) -> float:
  if len(points) < 2:
    return float(default)
  pts = sorted((float(k), float(v)) for (k, v) in points if float(k) > 0.0 and math.isfinite(float(v)))
  if len(pts) < 2:
    return float(default)
  kappa = float(kappa)
  if not (kappa > 0.0 and math.isfinite(kappa)):
    return float(default)
  if kappa <= pts[0][0]:
    return float(pts[0][1])
  if kappa >= pts[-1][0]:
    return float(pts[-1][1])
  xs = [math.log10(k) for (k, _v) in pts]
  ys = [v for (_k, v) in pts]
  return float(_interp_piecewise_linear(xs, ys, math.log10(kappa)))


def _extract_controller_constants(controller_path: str) -> dict:
  want = {
    "PHYSICS_A",
    "PHYSICS_B",
    "PHYSICS_C",
    "PHYSICS_D",
    "PHYSICS_MIN_LAT_ACCEL",
    "PHYSICS_MAX_LAT_ACCEL",
    "LOW_SPEED_BIAS_MPH",
    "LOW_SPEED_BIAS_END_MPH",
    "MAX_SPEED_DEFAULT",
    "SPEED_INCREASE_FACTOR",
  }
  src = _read_text(controller_path)
  tree = ast.parse(src, filename=controller_path)
  out: dict[str, float] = {}
  for node in tree.body:
    if not isinstance(node, ast.Assign):
      continue
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
      continue
    name = node.targets[0].id
    if name not in want:
      continue
    try:
      out[name] = float(ast.literal_eval(node.value))
    except Exception:
      continue
  missing = sorted(want - set(out.keys()))
  if missing:
    raise RuntimeError(f"failed to extract constants from {controller_path}: missing {missing}")
  return out


def _load_tuning_settings(tuning_path: str) -> tuple[bool, list[tuple[float, float]], dict]:
  try:
    src = _read_text(tuning_path)
  except FileNotFoundError:
    return False, [], {}

  tree = ast.parse(src, filename=tuning_path)
  enabled = False
  points: list[tuple[float, float]] = []
  meta: dict = {}

  for node in tree.body:
    if not isinstance(node, ast.Assign):
      continue
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
      continue
    name = node.targets[0].id
    if name == "Q_CURVE_ENABLED":
      try:
        enabled = bool(ast.literal_eval(node.value))
      except Exception:
        enabled = False
    elif name == "Q_CURVE_POINTS":
      try:
        raw = ast.literal_eval(node.value)
      except Exception:
        raw = []
      pts: list[tuple[float, float]] = []
      if isinstance(raw, list):
        for item in raw:
          if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
          try:
            k = float(item[0])
            q = float(item[1])
          except Exception:
            continue
          if not (k > 0.0 and math.isfinite(k) and math.isfinite(q)):
            continue
          pts.append((k, q))
      points = pts
    elif name == "Q_CURVE_META":
      try:
        raw = ast.literal_eval(node.value)
      except Exception:
        raw = {}
      if isinstance(raw, dict):
        meta = raw
  return enabled, points, meta


def _physics_based_lateral_acceleration(curvature: float, c: dict) -> float:
  curvature = max(1e-8, min(float(curvature), 1.0))
  result = c["PHYSICS_A"] / (1.0 + math.exp(c["PHYSICS_B"] * (curvature - c["PHYSICS_C"]))) + c["PHYSICS_D"]
  return _clip(result, c["PHYSICS_MIN_LAT_ACCEL"], c["PHYSICS_MAX_LAT_ACCEL"])


def base_curvature_to_speed_mps(abs_curvature_1pm: float, c: dict) -> float:
  if abs_curvature_1pm < 1e-7:
    return float(c["MAX_SPEED_DEFAULT"])

  safe_lat_accel = _physics_based_lateral_acceleration(abs_curvature_1pm, c)
  try:
    base_speed_mps = math.sqrt(safe_lat_accel / float(abs_curvature_1pm))
  except (ValueError, ZeroDivisionError):
    base_speed_mps = 0.0

  base_speed_mph = base_speed_mps * MS_TO_MPH
  if c["LOW_SPEED_BIAS_MPH"] != 0.0 and base_speed_mph < c["LOW_SPEED_BIAS_END_MPH"]:
    taper = 1.0 - (base_speed_mph / max(c["LOW_SPEED_BIAS_END_MPH"], 1e-3))
    base_speed_mph = base_speed_mph + c["LOW_SPEED_BIAS_MPH"] * _clip(taper, 0.0, 1.0)
    base_speed_mps = max(0.0, base_speed_mph * MPH_TO_MS)

  target_speed_mps = base_speed_mps * float(c["SPEED_INCREASE_FACTOR"])
  return _clip(target_speed_mps, 0.0, float(c["MAX_SPEED_DEFAULT"]))


def _sample_curve_kappa(kappa_max: float, kappa_min: float, n: int) -> list[float]:
  kappa = _logspace_desc(kappa_max, kappa_min, max(2, int(n)))
  # Ensure unique and strictly positive (log scale).
  out: list[float] = []
  last = None
  for k in kappa:
    k = max(1e-12, float(k))
    if last is None or abs(k - last) > 1e-12:
      out.append(k)
      last = k
  return out


def _format_kappa(kappa: float) -> str:
  if kappa <= 0.0 or not math.isfinite(kappa):
    return "n/a"
  if kappa >= 0.01:
    return f"{kappa:.3f}"
  if kappa >= 0.001:
    return f"{kappa:.4f}"
  return f"{kappa:.5f}"


def _format_speed(speed_mph: float) -> str:
  if not math.isfinite(speed_mph):
    return "n/a"
  return f"{speed_mph:.2f}"


# ===== California / Caltrans reference helpers =====
# These are for *tuning reference only* (not a safety guarantee, not legal advice).
#
# Caltrans HDM Figure 202.2 (Maximum Comfortable Speed on Horizontal Curves) provides:
#   e + f = 0.067 * V^2 / R
# with:
#   V in mph
#   R in feet
#   e = superelevation (ft/ft)
#   f = side friction factor (tabulated vs speed)
#
# This tuner uses that relationship as a rough comparative reference curve; the
# production VTSC mapping is still the baseline physics mapping + optional q(κ).
FT_PER_M = 3.28084
CALTRANS_HDM_FIG_202_2_C = 0.067
CALTRANS_HDM_FIG_202_2_SPEED_MPH = [20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
CALTRANS_HDM_FIG_202_2_SIDE_FRICTION_F = [0.27, 0.20, 0.16, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]
CALTRANS_HDM_E_DEFAULT = 0.06


def caltrans_hdm_fig_202_2_side_friction_factor(speed_mph: float) -> float:
  v = _clip(float(speed_mph), CALTRANS_HDM_FIG_202_2_SPEED_MPH[0], CALTRANS_HDM_FIG_202_2_SPEED_MPH[-1])
  return float(_interp_piecewise_linear(CALTRANS_HDM_FIG_202_2_SPEED_MPH, CALTRANS_HDM_FIG_202_2_SIDE_FRICTION_F, v))


def caltrans_hdm_fig_202_2_max_comfort_speed_mph(kappa_1pm: float, *, superelev_e: float = CALTRANS_HDM_E_DEFAULT) -> float:
  kappa = float(kappa_1pm)
  if not (kappa > 0.0 and math.isfinite(kappa)):
    return float("nan")

  r_m = 1.0 / max(kappa, 1e-12)
  r_ft = r_m * FT_PER_M
  if not (r_ft > 0.0 and math.isfinite(r_ft)):
    return float("nan")

  e = _clip(float(superelev_e), 0.0, 0.16)

  # Solve e + f(V) = 0.067*V^2/R via bisection (f(V) decreases with V, RHS increases with V^2).
  lo = 5.0
  hi = 100.0
  for _ in range(64):
    mid = 0.5 * (lo + hi)
    f = caltrans_hdm_fig_202_2_side_friction_factor(mid)
    g = e + f - (CALTRANS_HDM_FIG_202_2_C * mid * mid) / r_ft
    if g > 0.0:
      lo = mid
    else:
      hi = mid
  return 0.5 * (lo + hi)


@dataclass
class ControlPoint:
  kappa_1pm: float
  speed_mph: float


class CurveModel:
  def __init__(self, controller_path: str):
    self.controller_path = controller_path
    self.constants = _extract_controller_constants(controller_path)

  def base_speed_mps(self, kappa_1pm: float) -> float:
    return base_curvature_to_speed_mps(float(kappa_1pm), self.constants)

  def base_speed_mph(self, kappa_1pm: float) -> float:
    return self.base_speed_mps(kappa_1pm) * MS_TO_MPH

  def q_points_from_speed_points(self, speed_points: list[ControlPoint]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in speed_points:
      base_mph = self.base_speed_mph(p.kappa_1pm)
      q = 1.0 if base_mph <= 1e-6 else (float(p.speed_mph) / base_mph)
      if not math.isfinite(q):
        q = 1.0
      out.append((float(p.kappa_1pm), float(q)))
    return out

  def tuned_speed_mph(self, kappa_1pm: float, q_points: list[tuple[float, float]]) -> float:
    base_mph = self.base_speed_mph(kappa_1pm)
    q = _interp_logx(q_points, kappa_1pm, default=1.0)
    q = _clip(q, 0.5, 1.5)
    return base_mph * q


class SVGPlot:
  def __init__(self, *, width: int = 1100, height: int = 650):
    self.width = int(width)
    self.height = int(height)

  def _svg_header(self) -> list[str]:
    return [
      "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
      f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{self.width}\" height=\"{self.height}\" viewBox=\"0 0 {self.width} {self.height}\">",
      "<style>",
      "  .axis { stroke: #666; stroke-width: 2; fill: none; }",
      "  .grid { stroke: #333; stroke-width: 1; opacity: 0.35; }",
      "  .lbl { fill: #ccc; font-family: sans-serif; font-size: 14px; }",
      "  .title { fill: #fff; font-family: sans-serif; font-size: 22px; font-weight: 600; }",
      "  .curve_base { stroke: #aaa; stroke-width: 2; fill: none; opacity: 0.75; }",
      "  .curve_tuned { stroke: #27d3ff; stroke-width: 3; fill: none; }",
      "</style>",
      "<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#111\"/>",
    ]

  def _svg_footer(self) -> list[str]:
    return ["</svg>"]

  @staticmethod
  def _polyline(points: list[tuple[float, float]], cls: str) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in points)
    return f"<polyline class=\"{cls}\" points=\"{pts}\"/>"

  def write(
    self,
    path: str,
    *,
    title: str,
    kappa_desc: list[float],
    base_speed_mph: list[float],
    tuned_speed_mph: list[float] | None,
    y_max_mph: float,
    kappa_min: float,
    kappa_max: float,
    x_scale: str = "log",
  ) -> None:
    w = self.width
    h = self.height
    mx = 80
    my = 60
    plot_w = w - 2 * mx
    plot_h = h - 2 * my

    x_scale = str(x_scale or "log").lower().strip()
    if x_scale == "linear":
      denom = float(kappa_max) - float(kappa_min)
      if abs(denom) < 1e-12:
        denom = 1e-12

      def x_of_k(k: float) -> float:
        t = (float(kappa_max) - float(k)) / denom
        return mx + plot_w * _clip(t, 0.0, 1.0)
    else:
      log_min = math.log10(max(kappa_min, 1e-12))
      log_max = math.log10(max(kappa_max, 1e-12))
      denom = (log_min - log_max)
      if abs(denom) < 1e-9:
        denom = -1e-9 if denom < 0.0 else 1e-9

      def x_of_k(k: float) -> float:
        t = (math.log10(max(k, 1e-12)) - log_max) / denom
        return mx + plot_w * _clip(t, 0.0, 1.0)

    def y_of_v(v: float) -> float:
      t = float(v) / max(1e-6, float(y_max_mph))
      return my + plot_h * (1.0 - _clip(t, 0.0, 1.0))

    base_xy = [(x_of_k(k), y_of_v(v)) for (k, v) in zip(kappa_desc, base_speed_mph)]
    tuned_xy = None
    if tuned_speed_mph is not None:
      tuned_xy = [(x_of_k(k), y_of_v(v)) for (k, v) in zip(kappa_desc, tuned_speed_mph)]

    lines: list[str] = []
    lines.extend(self._svg_header())
    lines.append(f"<text class=\"title\" x=\"{mx}\" y=\"{my - 25}\">{title}</text>")

    # grid (speed)
    for mph in (20, 40, 60, 80):
      y = y_of_v(mph)
      lines.append(f"<line class=\"grid\" x1=\"{mx}\" y1=\"{y:.2f}\" x2=\"{mx + plot_w}\" y2=\"{y:.2f}\"/>")
      lines.append(f"<text class=\"lbl\" x=\"{mx - 55}\" y=\"{y + 5:.2f}\">{mph} mph</text>")

    # axes
    lines.append(f"<line class=\"axis\" x1=\"{mx}\" y1=\"{my}\" x2=\"{mx}\" y2=\"{my + plot_h}\"/>")
    lines.append(f"<line class=\"axis\" x1=\"{mx}\" y1=\"{my + plot_h}\" x2=\"{mx + plot_w}\" y2=\"{my + plot_h}\"/>")
    lines.append(f"<text class=\"lbl\" x=\"{mx}\" y=\"{h - 15}\">curvature κ (1/m): left=tight, right=straight</text>")

    lines.append(self._polyline(base_xy, "curve_base"))
    if tuned_xy is not None:
      lines.append(self._polyline(tuned_xy, "curve_tuned"))

    lines.extend(self._svg_footer())
    _write_text(path, "\n".join(lines) + "\n")


class CurveTunerApp:
  def __init__(
    self,
    root: tk.Tk,
    *,
    controller_path: str,
    tuning_path: str,
    history_path: str,
  ):
    self.root = root
    self.controller_path = controller_path
    self.tuning_path = tuning_path
    self.history_path = history_path
    self.model = CurveModel(controller_path)

    self.kappa_max = 2.0e-2
    self.kappa_min = 1.0e-5
    self.y_max_mph = 90.0
    self.n_points = 12
    self.n_samples = 250

    self.lock_x = tk.BooleanVar(value=True)
    self.enforce_monotonic = tk.BooleanVar(value=True)
    self.x_scale_var = tk.StringVar(value="log")
    self.edit_mode_var = tk.StringVar(value="mph")
    self.point_count_var = tk.IntVar(value=int(self.n_points))
    self.ref_visible = tk.BooleanVar(value=False)

    self.note_var = tk.StringVar(value="")
    self.status_var = tk.StringVar(value="")
    self.code_enabled_var = tk.BooleanVar(value=False)

    self.selected_idx: int | None = None
    self.dragging = False
    self._drag_dx = 0.0
    self._drag_dy = 0.0

    self.points: list[ControlPoint] = []
    self.history: list[dict] = _load_jsonl(history_path)
    self._curve_cache_key = None
    self._curve_cache = None
    self._ref_after = None
    self._build_ui()
    self._refresh_history_list()
    self._reset_points_to_baseline(redraw=False)
    self._load_current_config(auto=True)
    self._redraw()

  def _build_ui(self) -> None:
    self.root.title("VTSC Curve Tuner")

    outer = ttk.Frame(self.root)
    outer.pack(fill="both", expand=True)

    left = ttk.Frame(outer, padding=10)
    left.pack(side="left", fill="y")

    right = ttk.Frame(outer, padding=10)
    right.pack(side="right", fill="both", expand=True)

    # History
    ttk.Label(left, text="History").pack(anchor="w")
    self.history_list = tk.Listbox(left, width=45, height=18)
    self.history_list.pack(fill="y", expand=False)
    self.history_list.bind("<<ListboxSelect>>", self._on_history_select)

    hist_btns = ttk.Frame(left)
    hist_btns.pack(fill="x", pady=(6, 0))
    ttk.Button(hist_btns, text="Load", command=self._load_selected_history).pack(side="left")
    ttk.Button(hist_btns, text="Load Current", command=self._load_current_config).pack(side="left", padx=6)
    ttk.Button(hist_btns, text="Export SVG", command=self._export_svg_dialog).pack(side="left", padx=6)
    ttk.Button(hist_btns, text="Apply", command=self._apply_to_code).pack(side="left")

    ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

    # Presets
    ttk.Label(left, text="Preset generator").pack(anchor="w")
    self.preset_var = tk.StringVar(value="Smooth bump")
    preset = ttk.Combobox(left, textvariable=self.preset_var, state="readonly", values=[
      "Smooth bump",
      "Exponential bump",
      "Gaussian bump",
      "Sigmoid tilt",
      "Reset (q=1)",
    ])
    preset.pack(fill="x")

    frm = ttk.Frame(left)
    frm.pack(fill="x", pady=(6, 0))
    ttk.Label(frm, text="center κ").grid(row=0, column=0, sticky="w")
    ttk.Label(frm, text="target mph").grid(row=1, column=0, sticky="w")
    ttk.Label(frm, text="width (decades)").grid(row=2, column=0, sticky="w")
    self.center_kappa_var = tk.StringVar(value="0.003924")
    self.target_mph_var = tk.StringVar(value="70.0")
    self.width_decades_var = tk.StringVar(value="0.55")
    ttk.Entry(frm, textvariable=self.center_kappa_var, width=16).grid(row=0, column=1, sticky="ew")
    ttk.Entry(frm, textvariable=self.target_mph_var, width=16).grid(row=1, column=1, sticky="ew")
    ttk.Entry(frm, textvariable=self.width_decades_var, width=16).grid(row=2, column=1, sticky="ew")
    frm.columnconfigure(1, weight=1)
    ttk.Button(left, text="Generate", command=self._generate_preset).pack(fill="x", pady=(6, 0))

    ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

    # Curve points
    ttk.Label(left, text="Curve points").pack(anchor="w")
    pts = ttk.Frame(left)
    pts.pack(fill="x", pady=(4, 0))
    ttk.Label(pts, text="count").grid(row=0, column=0, sticky="w")
    try:
      spin = ttk.Spinbox(pts, from_=2, to=40, textvariable=self.point_count_var, width=6)
    except Exception:
      spin = tk.Spinbox(pts, from_=2, to=40, textvariable=self.point_count_var, width=6)
    spin.grid(row=0, column=1, sticky="w")
    ttk.Button(pts, text="Resample", command=self._resample_points_from_ui).grid(row=0, column=2, padx=6, sticky="w")
    pts.columnconfigure(3, weight=1)

    pt_btns = ttk.Frame(left)
    pt_btns.pack(fill="x", pady=(6, 0))
    ttk.Button(pt_btns, text="Add point", command=self._add_point_button).pack(side="left", fill="x", expand=True)
    ttk.Button(pt_btns, text="Delete point", command=self._delete_selected_point).pack(side="left", fill="x", expand=True, padx=6)

    ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

    # Selected point editor
    ttk.Label(left, text="Selected point").pack(anchor="w")
    sel = ttk.Frame(left)
    sel.pack(fill="x", pady=(4, 0))
    ttk.Label(sel, text="κ").grid(row=0, column=0, sticky="w")
    ttk.Label(sel, text="mph").grid(row=1, column=0, sticky="w")
    ttk.Label(sel, text="q").grid(row=2, column=0, sticky="w")
    self.sel_kappa_var = tk.StringVar(value="")
    self.sel_mph_var = tk.StringVar(value="")
    self.sel_q_var = tk.StringVar(value="")
    ttk.Entry(sel, textvariable=self.sel_kappa_var, width=16).grid(row=0, column=1, sticky="ew")
    ttk.Entry(sel, textvariable=self.sel_mph_var, width=16).grid(row=1, column=1, sticky="ew")
    ttk.Entry(sel, textvariable=self.sel_q_var, width=16).grid(row=2, column=1, sticky="ew")
    sel.columnconfigure(1, weight=1)
    mode = ttk.Frame(left)
    mode.pack(fill="x", pady=(4, 0))
    ttk.Label(mode, text="Edit using:").pack(side="left")
    ttk.Radiobutton(mode, text="mph", variable=self.edit_mode_var, value="mph").pack(side="left", padx=6)
    ttk.Radiobutton(mode, text="q", variable=self.edit_mode_var, value="q").pack(side="left")
    ttk.Button(left, text="Update point", command=self._update_selected_point_from_fields).pack(fill="x", pady=(6, 0))

    ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

    ttk.Checkbutton(left, text="Lock X (Shift=horizontal)", variable=self.lock_x, command=self._redraw).pack(anchor="w")
    ttk.Checkbutton(left, text="Enforce monotonic", variable=self.enforce_monotonic, command=self._on_toggle_monotonic).pack(anchor="w")
    ttk.Checkbutton(left, text="Enable tuning in code (on Apply)", variable=self.code_enabled_var).pack(anchor="w")

    ttk.Label(left, text="X-axis scale").pack(anchor="w", pady=(10, 0))
    scale = ttk.Frame(left)
    scale.pack(fill="x")
    ttk.Radiobutton(scale, text="log", variable=self.x_scale_var, value="log", command=self._redraw).pack(side="left")
    ttk.Radiobutton(scale, text="linear", variable=self.x_scale_var, value="linear", command=self._redraw).pack(side="left", padx=6)

    ttk.Label(left, text="Note (stored in history + apply)").pack(anchor="w", pady=(10, 0))
    ttk.Entry(left, textvariable=self.note_var).pack(fill="x")

    ttk.Button(left, text="Reset to baseline", command=self._reset_points_to_baseline).pack(fill="x", pady=(6, 0))

    # Right-side: plot + optional reference panel
    right_top = ttk.Frame(right)
    right_top.pack(fill="x")
    self.ref_toggle_btn = ttk.Button(right_top, text="Show Reference ▶", command=self._toggle_reference_panel)
    self.ref_toggle_btn.pack(side="right")

    right_body = ttk.Frame(right)
    right_body.pack(fill="both", expand=True)

    self.plot_frame = ttk.Frame(right_body)
    self.plot_frame.pack(side="left", fill="both", expand=True)

    # Plot canvas
    self.canvas = tk.Canvas(self.plot_frame, width=920, height=560, bg="#111111", highlightthickness=0)
    self.canvas.pack(fill="both", expand=True)
    self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
    self.canvas.bind("<Double-Button-1>", self._on_double_click)
    self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
    self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
    self.canvas.bind("<Configure>", self._on_canvas_configure)

    status = ttk.Frame(self.plot_frame)
    status.pack(fill="x", pady=(8, 0))
    ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

    self.ref_frame = ttk.Frame(right_body, width=380)
    ref_hdr = ttk.Frame(self.ref_frame)
    ref_hdr.pack(fill="x")
    ttk.Label(ref_hdr, text="California reference (for tuning)").pack(side="left")
    ttk.Button(ref_hdr, text="Hide ◀", command=self._toggle_reference_panel).pack(side="right")

    ref_body = ttk.Frame(self.ref_frame)
    ref_body.pack(fill="both", expand=True, pady=(6, 0))
    self.ref_text = tk.Text(ref_body, width=44, height=32, wrap="word", bg="#0f0f0f", fg="#dddddd", relief="flat")
    ref_scroll = ttk.Scrollbar(ref_body, orient="vertical", command=self.ref_text.yview)
    self.ref_text.configure(yscrollcommand=ref_scroll.set)
    self.ref_text.pack(side="left", fill="both", expand=True)
    ref_scroll.pack(side="right", fill="y")

  def _canvas_dims(self) -> tuple[float, float]:
    # During initial startup Tk may report width/height=1 until layout is settled.
    w = float(self.canvas.winfo_width() or 0.0)
    h = float(self.canvas.winfo_height() or 0.0)
    if w < 300.0:
      w = 920.0
    if h < 250.0:
      h = 560.0
    return w, h

  def _kappa_to_x(self, kappa_1pm: float) -> float:
    w, _h = self._canvas_dims()
    mx = 80.0
    plot_w = max(1.0, w - 2.0 * mx)
    if self.x_scale_var.get() == "linear":
      denom = float(self.kappa_max) - float(self.kappa_min)
      if abs(denom) < 1e-12:
        denom = 1e-12
      t = (float(self.kappa_max) - float(kappa_1pm)) / denom
    else:
      log_min = math.log10(max(self.kappa_min, 1e-12))
      log_max = math.log10(max(self.kappa_max, 1e-12))
      denom = (log_min - log_max)
      if abs(denom) < 1e-9:
        denom = -1e-9 if denom < 0.0 else 1e-9
      t = (math.log10(max(float(kappa_1pm), 1e-12)) - log_max) / denom
    return mx + plot_w * _clip(t, 0.0, 1.0)

  def _x_to_kappa(self, x: float) -> float:
    w, _h = self._canvas_dims()
    mx = 80.0
    plot_w = max(1.0, w - 2.0 * mx)
    t = (float(x) - mx) / plot_w
    t = _clip(t, 0.0, 1.0)
    if self.x_scale_var.get() == "linear":
      return float(self.kappa_max) + t * (float(self.kappa_min) - float(self.kappa_max))
    log_min = math.log10(max(self.kappa_min, 1e-12))
    log_max = math.log10(max(self.kappa_max, 1e-12))
    log_k = log_max + t * (log_min - log_max)
    return float(10.0 ** log_k)

  def _speed_to_y(self, speed_mph: float) -> float:
    _w, h = self._canvas_dims()
    my = 60.0
    plot_h = max(1.0, h - 2.0 * my)
    t = float(speed_mph) / max(1e-6, float(self.y_max_mph))
    return my + plot_h * (1.0 - _clip(t, 0.0, 1.0))

  def _y_to_speed(self, y: float) -> float:
    _w, h = self._canvas_dims()
    my = 60.0
    plot_h = max(1.0, h - 2.0 * my)
    t = 1.0 - ((float(y) - my) / plot_h)
    t = _clip(t, 0.0, 1.0)
    return float(self.y_max_mph) * t

  def _reset_points_to_baseline(self, *, redraw: bool = True) -> None:
    try:
      self.n_points = int(_clip(float(self.point_count_var.get()), 2.0, 40.0))
    except Exception:
      self.n_points = int(_clip(float(self.n_points), 2.0, 40.0))
    self.point_count_var.set(int(self.n_points))
    kappas = _logspace_desc(self.kappa_max, self.kappa_min, int(self.n_points))
    self.points = []
    for k in kappas:
      self.points.append(ControlPoint(kappa_1pm=float(k), speed_mph=float(self.model.base_speed_mph(k))))
    self.selected_idx = None
    self.status_var.set("Reset points to baseline")
    if redraw:
      self._redraw()

  def _q_points(self) -> list[tuple[float, float]]:
    # Clamp q so Apply never writes out-of-range values to the tuning file.
    out: list[tuple[float, float]] = []
    for k, q in self.model.q_points_from_speed_points(self.points):
      out.append((float(k), _clip(float(q), 0.5, 1.5)))
    return out

  def _set_points(self, points: list[ControlPoint], *, selected_idx: int | None = None, status: str | None = None) -> None:
    pts = []
    for p in points:
      k = _clip(float(p.kappa_1pm), float(self.kappa_min), float(self.kappa_max))
      mph = _clip(float(p.speed_mph), 0.0, float(self.y_max_mph))
      if not (k > 0.0 and math.isfinite(k) and math.isfinite(mph)):
        continue
      pts.append(ControlPoint(kappa_1pm=k, speed_mph=mph))
    pts.sort(key=lambda p: p.kappa_1pm, reverse=True)
    if len(pts) < 2:
      self._reset_points_to_baseline()
      return

    self.points = pts
    self.n_points = int(len(self.points))
    self.point_count_var.set(int(self.n_points))

    if self.enforce_monotonic.get():
      self._enforce_monotonic()

    self.selected_idx = selected_idx
    self._update_selected_fields()
    if status:
      self.status_var.set(status)
    self._redraw()

  def _tuned_speed_mph_at(self, kappa_1pm: float) -> float:
    q_pts = self._q_points()
    return float(self.model.tuned_speed_mph(float(kappa_1pm), q_pts))

  def _insert_point(self, kappa_1pm: float, speed_mph: float) -> None:
    kappa = _clip(float(kappa_1pm), float(self.kappa_min), float(self.kappa_max))
    mph = _clip(float(speed_mph), 0.0, float(self.y_max_mph))

    # Avoid duplicate κ values (log interpolation & ordering are cleaner with uniqueness).
    for _ in range(12):
      clash = False
      for p in self.points:
        if abs(p.kappa_1pm - kappa) <= max(1e-12, 1e-6 * kappa):
          clash = True
          kappa = _clip(kappa * 0.997, float(self.kappa_min), float(self.kappa_max))
          break
      if not clash:
        break

    insert_at = len(self.points)
    for i, p in enumerate(self.points):
      if kappa > p.kappa_1pm:
        insert_at = i
        break

    new_pts = list(self.points)
    new_pts.insert(insert_at, ControlPoint(kappa_1pm=kappa, speed_mph=mph))
    self._set_points(new_pts, selected_idx=insert_at, status="Added point")

  def _add_point_button(self) -> None:
    if not self.points:
      self._reset_points_to_baseline()
      return

    if self.selected_idx is None:
      kappa = _safe_float(self.center_kappa_var.get(), 0.003924)
      mph = self._tuned_speed_mph_at(kappa)
      self._insert_point(kappa, mph)
      return

    idx = int(self.selected_idx)
    left_k = self.points[idx].kappa_1pm
    right_k = None
    if idx < len(self.points) - 1:
      right_k = self.points[idx + 1].kappa_1pm
    elif idx > 0:
      right_k = self.points[idx - 1].kappa_1pm

    if right_k is None:
      kappa = _safe_float(self.center_kappa_var.get(), left_k)
    else:
      # Midpoint in log space (geometric mean) feels natural for curvature.
      kappa = math.sqrt(max(left_k, 1e-12) * max(right_k, 1e-12))
    mph = self._tuned_speed_mph_at(kappa)
    self._insert_point(kappa, mph)

  def _delete_selected_point(self) -> None:
    if self.selected_idx is None:
      return
    if len(self.points) <= 2:
      self.status_var.set("Cannot delete: keep at least 2 points")
      return
    idx = int(self.selected_idx)
    new_pts = list(self.points)
    new_pts.pop(idx)
    new_idx = min(idx, len(new_pts) - 1)
    self._set_points(new_pts, selected_idx=new_idx, status="Deleted point")

  def _resample_points_from_ui(self) -> None:
    try:
      n = int(_clip(float(self.point_count_var.get()), 2.0, 40.0))
    except Exception:
      return
    self._resample_points(n)

  def _resample_points(self, n_points: int) -> None:
    n = int(_clip(float(n_points), 2.0, 40.0))
    if not self.points:
      self.n_points = n
      self.point_count_var.set(n)
      self._reset_points_to_baseline()
      return

    q_pts = self._q_points()
    if self.x_scale_var.get() == "linear":
      kappas = []
      for i in range(n):
        t = i / float(n - 1)
        kappas.append(float(self.kappa_max) + t * (float(self.kappa_min) - float(self.kappa_max)))
    else:
      kappas = _logspace_desc(self.kappa_max, self.kappa_min, n)

    new_pts: list[ControlPoint] = []
    for k in kappas:
      mph = self.model.tuned_speed_mph(float(k), q_pts)
      new_pts.append(ControlPoint(kappa_1pm=float(k), speed_mph=float(mph)))

    self._set_points(new_pts, selected_idx=None, status=f"Resampled to {n} points")

  def _on_double_click(self, ev: tk.Event) -> None:
    # Add a point at the cursor location (useful for targeted bumps).
    kappa = self._x_to_kappa(ev.x)
    mph = self._y_to_speed(ev.y)
    self._insert_point(kappa, mph)

  def _toggle_reference_panel(self) -> None:
    show = not bool(self.ref_visible.get())
    self.ref_visible.set(show)
    if show:
      self.ref_frame.pack(side="right", fill="y", padx=(10, 0))
      self.ref_toggle_btn.configure(text="Hide Reference ◀")
      self._update_reference_text()
    else:
      try:
        self.ref_frame.pack_forget()
      except Exception:
        pass
      self.ref_toggle_btn.configure(text="Show Reference ▶")
    self._redraw()

  def _update_reference_text(self) -> None:
    if not bool(self.ref_visible.get()):
      return
    lines: list[str] = []
    lines.append("Caltrans Highway Design Manual (HDM) – Fig 202.2 (tuning reference)")
    lines.append("  e + f = 0.067 * V^2 / R")
    lines.append("  V: mph   R: feet   e: superelevation (ft/ft)   f: side friction factor")
    lines.append("  f decreases with speed (roughly ~0.27@20mph → ~0.08@80mph).")
    lines.append("  Note: Fig 202.2 is an *aid* for comfortable speeds; HDM standards live in the 202.2 tables.")
    lines.append("")
    lines.append("California MUTCD / MUTCD Advisory Speed Plaque (general guidance)")
    lines.append("  Advisory speeds are set by an engineering study (not directly by curvature).")
    lines.append("  A common method uses ball-bank criteria: 16° (≤20mph), 14° (25–30mph), 12° (≥35mph).")
    lines.append("  Advisory plaques are typically in 5 mph increments.")
    lines.append("")

    if self.selected_idx is not None:
      p = self.points[self.selected_idx]
      kappa = float(p.kappa_1pm)
      r_m = 1.0 / max(kappa, 1e-12)
      r_ft = r_m * FT_PER_M
      cal_e6 = caltrans_hdm_fig_202_2_max_comfort_speed_mph(kappa, superelev_e=0.06)
      cal_e8 = caltrans_hdm_fig_202_2_max_comfort_speed_mph(kappa, superelev_e=0.08)
      lines.append("Selected point")
      lines.append(f"  κ={_format_kappa(kappa)}  (R≈{r_ft:.0f} ft / {r_m:.0f} m)")
      lines.append(f"  tuned={_format_speed(p.speed_mph)} mph")
      if math.isfinite(cal_e6):
        lines.append(f"  HDM Fig 202.2 ref: ~{cal_e6:.1f} mph @ e=0.06, ~{cal_e8:.1f} mph @ e=0.08")

    self.ref_text.configure(state="normal")
    self.ref_text.delete("1.0", tk.END)
    self.ref_text.insert("1.0", "\n".join(lines) + "\n")
    self.ref_text.configure(state="disabled")

  def _schedule_reference_update(self, delay_ms: int = 120) -> None:
    if not bool(self.ref_visible.get()):
      return
    try:
      if self._ref_after is not None:
        self.root.after_cancel(self._ref_after)
    except Exception:
      pass
    self._ref_after = self.root.after(int(delay_ms), self._update_reference_text)

  def _curve_samples(self) -> tuple[list[float], list[float]]:
    key = (
      str(self.x_scale_var.get()),
      int(self.n_samples),
      float(self.kappa_min),
      float(self.kappa_max),
    )
    if self._curve_cache_key == key and self._curve_cache is not None:
      return self._curve_cache

    x_scale = str(self.x_scale_var.get() or "log").strip().lower()
    n = max(50, int(self.n_samples))
    if x_scale == "linear":
      kappas = []
      for i in range(n):
        t = i / float(n - 1)
        kappas.append(float(self.kappa_max) + t * (float(self.kappa_min) - float(self.kappa_max)))
    else:
      kappas = _sample_curve_kappa(self.kappa_max, self.kappa_min, n)
    base_mph = [self.model.base_speed_mph(k) for k in kappas]
    self._curve_cache_key = key
    self._curve_cache = (kappas, base_mph)
    return self._curve_cache

  @staticmethod
  def _tuned_curve_from_q(kappas: list[float], base_mph: list[float], q_points: list[tuple[float, float]]) -> list[float]:
    if len(q_points) < 2:
      return list(base_mph)
    pts = sorted((float(k), float(q)) for (k, q) in q_points if float(k) > 0.0 and math.isfinite(float(q)))
    if len(pts) < 2:
      return list(base_mph)
    xs = [math.log10(max(k, 1e-12)) for (k, _q) in pts]
    ys = [q for (_k, q) in pts]
    tuned = []
    for k, base in zip(kappas, base_mph):
      q = float(_interp_piecewise_linear(xs, ys, math.log10(max(float(k), 1e-12))))
      q = _clip(q, 0.5, 1.5)
      tuned.append(float(base) * q)
    return tuned

  def _enforce_monotonic(self) -> None:
    # Display order is left->right: high curvature -> low curvature, so speeds should be non-decreasing.
    last = None
    for i, p in enumerate(self.points):
      if last is None:
        last = float(p.speed_mph)
        continue
      if p.speed_mph < last:
        self.points[i] = ControlPoint(kappa_1pm=p.kappa_1pm, speed_mph=last)
      else:
        last = float(p.speed_mph)

  def _on_toggle_monotonic(self) -> None:
    if self.enforce_monotonic.get():
      self._enforce_monotonic()
    self._redraw()

  def _generate_preset(self) -> None:
    preset = self.preset_var.get()
    if preset == "Reset (q=1)":
      self._reset_points_to_baseline()
      return

    center_kappa = _safe_float(self.center_kappa_var.get(), 0.003924)
    target_mph = _safe_float(self.target_mph_var.get(), 70.0)
    width_decades = _safe_float(self.width_decades_var.get(), 0.55)
    width_decades = max(0.05, min(width_decades, 2.0))

    base_center = self.model.base_speed_mph(center_kappa)
    q_center = 1.0 if base_center <= 1e-6 else (target_mph / base_center)
    q_center = _clip(q_center, 0.5, 1.5)

    log_center = math.log10(max(center_kappa, 1e-12))
    half_w = 0.5 * width_decades

    new_points: list[ControlPoint] = []
    for p in self.points:
      base_mph = self.model.base_speed_mph(p.kappa_1pm)
      if base_mph <= 1e-6:
        new_points.append(ControlPoint(kappa_1pm=p.kappa_1pm, speed_mph=base_mph))
        continue
      d = abs(math.log10(max(p.kappa_1pm, 1e-12)) - log_center)
      if preset == "Gaussian bump":
        # Rough gaussian in log space; sigma from width.
        sigma = max(0.05, half_w / 1.177)
        w = math.exp(-0.5 * (d / sigma) ** 2)
      elif preset == "Exponential bump":
        # Exponential "bump" in log space (Laplace-like): w=0.5 at d=half_w.
        tau = max(0.05, half_w / math.log(2.0))
        w = math.exp(-d / tau)
      elif preset == "Sigmoid tilt":
        # A gentle "tilt": raise tighter curves more than straights.
        # Map log-kappa distance to a 0..1 ramp then sigmoid-ish it.
        t = 1.0 - _clip((math.log10(max(p.kappa_1pm, 1e-12)) - (log_center - half_w)) / max(1e-6, width_decades), 0.0, 1.0)
        w = _smoothstep(t)
      else:
        # Smooth bump (cosmetic EQ-like window)
        if d >= half_w:
          w = 0.0
        else:
          w = 1.0 - (d / max(1e-6, half_w))
          w = _smoothstep(w)
      q = 1.0 + (q_center - 1.0) * w
      q = _clip(q, 0.5, 1.5)
      new_points.append(ControlPoint(kappa_1pm=p.kappa_1pm, speed_mph=base_mph * q))

    self.points = new_points
    if self.enforce_monotonic.get():
      self._enforce_monotonic()
    self.status_var.set(f"Generated preset: {preset} (center κ={_format_kappa(center_kappa)}, target={_format_speed(target_mph)} mph)")
    self._redraw()

  def _load_current_config(self, *, auto: bool = False) -> None:
    enabled, q_points, meta = _load_tuning_settings(self.tuning_path)
    self.code_enabled_var.set(bool(enabled))
    note = str(meta.get("note", "")).strip()
    if auto and self.note_var.get().strip() == "" and note:
      self.note_var.set(note)

    if len(q_points) < 2:
      self.status_var.set("Loaded current config: baseline. Tip: double-click the plot to add a point; uncheck Lock X to drag horizontally.")
      if not auto:
        self._reset_points_to_baseline()
      return

    meta_note = f" ({note})" if note else ""
    pts: list[ControlPoint] = []
    for k, q in q_points:
      base_mph = self.model.base_speed_mph(float(k))
      q = _clip(float(q), 0.5, 1.5)
      pts.append(ControlPoint(kappa_1pm=float(k), speed_mph=float(base_mph * q)))
    pts.sort(key=lambda p: p.kappa_1pm, reverse=True)
    self._set_points(pts, selected_idx=None, status=f"Loaded current config: q-curve ({len(q_points)} pts, enabled={enabled}){meta_note}")

  def _redraw(self) -> None:
    c = self.canvas
    c.delete("all")

    w, h = self._canvas_dims()
    mx = 80.0
    my = 60.0
    plot_w = max(1.0, w - 2.0 * mx)
    plot_h = max(1.0, h - 2.0 * my)

    # axes + grid
    c.create_rectangle(0, 0, w, h, fill="#111111", outline="")
    c.create_line(mx, my, mx, my + plot_h, fill="#666666", width=2)
    c.create_line(mx, my + plot_h, mx + plot_w, my + plot_h, fill="#666666", width=2)

    for mph in (20, 40, 60, 80):
      y = self._speed_to_y(mph)
      c.create_line(mx, y, mx + plot_w, y, fill="#2a2a2a", width=1)
      c.create_text(mx - 45, y, text=f"{mph} mph", fill="#cccccc", font=("TkDefaultFont", 10))

    # Curvature tick labels
    if self.x_scale_var.get() == "linear":
      n_ticks = 6
      for i in range(n_ticks):
        t = i / float(n_ticks - 1)
        k = float(self.kappa_max) + t * (float(self.kappa_min) - float(self.kappa_max))
        x = self._kappa_to_x(k)
        c.create_line(x, my + plot_h, x, my + plot_h + 6, fill="#666666", width=1)
        c.create_text(x, my + plot_h + 18, text=_format_kappa(k), fill="#aaaaaa", font=("TkDefaultFont", 9))
    else:
      for k in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001):
        if k < self.kappa_min or k > self.kappa_max:
          continue
        x = self._kappa_to_x(k)
        c.create_line(x, my + plot_h, x, my + plot_h + 6, fill="#666666", width=1)
        c.create_text(x, my + plot_h + 18, text=_format_kappa(k), fill="#aaaaaa", font=("TkDefaultFont", 9))

    c.create_text(mx, my - 18, text="curvature→speed (baseline vs tuned)", fill="#ffffff", font=("TkDefaultFont", 14, "bold"), anchor="w")
    c.create_text(
      mx,
      h - 14,
      text="drag: adjust point • Shift=horizontal • double-click: add point • Lock X: EQ-style",
      fill="#8a8a8a",
      font=("TkDefaultFont", 9),
      anchor="w",
    )

    # Curves (cached baseline; tuned recomputed from q-points)
    kappas, base_mph = self._curve_samples()
    q_pts = self._q_points()
    tuned_mph = self._tuned_curve_from_q(kappas, base_mph, q_pts)

    def draw_polyline(values_mph: list[float], color: str, width_px: int, dash: tuple[int, int] | None = None) -> None:
      pts: list[float] = []
      for k, v in zip(kappas, values_mph):
        pts.append(self._kappa_to_x(k))
        pts.append(self._speed_to_y(v))
      if len(pts) >= 4:
        c.create_line(*pts, fill=color, width=width_px, smooth=True, splinesteps=24, dash=dash)

    draw_polyline(base_mph, "#aaaaaa", 2, dash=(4, 4))
    draw_polyline(tuned_mph, "#27d3ff", 3)

    # Points
    for i, p in enumerate(self.points):
      x = self._kappa_to_x(p.kappa_1pm)
      y = self._speed_to_y(p.speed_mph)
      r = 6
      fill = "#ffcc00" if self.selected_idx == i else "#27d3ff"
      c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="#000000")

    # status based on selection
    if self.selected_idx is not None:
      p = self.points[self.selected_idx]
      base = self.model.base_speed_mph(p.kappa_1pm)
      q = 1.0 if base <= 1e-6 else (p.speed_mph / base)
      self.status_var.set(
        f"idx={self.selected_idx}  κ={_format_kappa(p.kappa_1pm)}  base={_format_speed(base)} mph  tuned={_format_speed(p.speed_mph)} mph  q={_format_speed(q)}"
      )

  def _on_canvas_configure(self, _ev: tk.Event) -> None:
    # Redraw on resize with a light debounce to avoid flicker on live-resize.
    try:
      after_id = getattr(self, "_resize_after", None)
      if after_id is not None:
        self.root.after_cancel(after_id)
    except Exception:
      pass
    self._resize_after = self.root.after(60, self._redraw)

  def _nearest_point_idx(self, x: float, y: float) -> int | None:
    best = None
    best_d2 = None
    for i, p in enumerate(self.points):
      px = self._kappa_to_x(p.kappa_1pm)
      py = self._speed_to_y(p.speed_mph)
      d2 = (px - x) ** 2 + (py - y) ** 2
      if best_d2 is None or d2 < best_d2:
        best = i
        best_d2 = d2
    if best_d2 is not None and best_d2 <= (14.0 ** 2):
      return best
    return None

  def _on_mouse_down(self, ev: tk.Event) -> None:
    try:
      self.canvas.focus_set()
    except Exception:
      pass
    idx = self._nearest_point_idx(ev.x, ev.y)
    self.selected_idx = idx
    self.dragging = idx is not None
    self._drag_dx = 0.0
    self._drag_dy = 0.0
    if idx is not None:
      p = self.points[idx]
      px = self._kappa_to_x(p.kappa_1pm)
      py = self._speed_to_y(p.speed_mph)
      self._drag_dx = float(ev.x) - float(px)
      self._drag_dy = float(ev.y) - float(py)
    if self.dragging:
      try:
        self.canvas.grab_set()
      except Exception:
        pass
    self._update_selected_fields()
    self._redraw()

  def _on_mouse_drag(self, ev: tk.Event) -> None:
    if not self.dragging or self.selected_idx is None:
      return
    idx = self.selected_idx
    p = self.points[idx]

    # Tk state bit 0x0001 is Shift; allow Shift-drag to move horizontally even when "Lock X" is enabled.
    allow_x = (not self.lock_x.get()) or (int(getattr(ev, "state", 0)) & 0x0001) != 0

    target_x = float(ev.x) - float(self._drag_dx)
    target_y = float(ev.y) - float(self._drag_dy)

    new_kappa = float(p.kappa_1pm)
    if allow_x:
      new_kappa = _clip(self._x_to_kappa(target_x), self.kappa_min, self.kappa_max)
      # Keep ordering (left->right decreasing kappa): neighbors constrain.
      if idx > 0:
        left = self.points[idx - 1].kappa_1pm
        new_kappa = min(new_kappa, left * 0.999)
      if idx < len(self.points) - 1:
        right = self.points[idx + 1].kappa_1pm
        new_kappa = max(new_kappa, right * 1.001)

    # Dragging uses the y-axis (mph), but clamp via q(κ) to keep tuning sane and consistent with runtime.
    desired_mph = _clip(self._y_to_speed(target_y), 0.0, self.y_max_mph)
    base_mph = float(self.model.base_speed_mph(new_kappa))
    q = 1.0 if base_mph <= 1e-6 else (desired_mph / base_mph)
    q = _clip(float(q), 0.5, 1.5)
    new_speed = _clip(base_mph * q, 0.0, self.y_max_mph)
    self.points[idx] = ControlPoint(kappa_1pm=new_kappa, speed_mph=new_speed)
    self._update_selected_fields()
    self._redraw()

  def _on_mouse_up(self, _ev: tk.Event) -> None:
    self.dragging = False
    try:
      self.canvas.grab_release()
    except Exception:
      pass
    if self.enforce_monotonic.get():
      self._enforce_monotonic()
      self._update_selected_fields()
      self._redraw()
    self._schedule_reference_update(delay_ms=0)

  def _update_selected_fields(self) -> None:
    if self.selected_idx is None:
      self.sel_kappa_var.set("")
      self.sel_mph_var.set("")
      self.sel_q_var.set("")
      if not self.dragging:
        self._schedule_reference_update()
      return
    p = self.points[self.selected_idx]
    base = self.model.base_speed_mph(p.kappa_1pm)
    q = 1.0 if base <= 1e-6 else (p.speed_mph / base)
    self.sel_kappa_var.set(_format_kappa(p.kappa_1pm))
    self.sel_mph_var.set(_format_speed(p.speed_mph))
    self.sel_q_var.set(_format_speed(q))
    if not self.dragging:
      self._schedule_reference_update()

  def _update_selected_point_from_fields(self) -> None:
    if self.selected_idx is None:
      return
    idx = self.selected_idx
    p = self.points[idx]

    kappa = _safe_float(self.sel_kappa_var.get(), p.kappa_1pm)
    kappa = _clip(kappa, self.kappa_min, self.kappa_max)
    base = self.model.base_speed_mph(kappa)

    mode = str(self.edit_mode_var.get() or "mph").strip().lower()
    if mode == "q":
      q = _safe_float(self.sel_q_var.get(), 1.0 if base <= 1e-6 else (p.speed_mph / base))
      q = _clip(q, 0.5, 1.5)
      mph = base * q
    else:
      mph = _safe_float(self.sel_mph_var.get(), p.speed_mph)
    kappa = _clip(kappa, self.kappa_min, self.kappa_max)
    mph = _clip(mph, 0.0, self.y_max_mph)

    new_pts = list(self.points)
    new_pts[idx] = ControlPoint(kappa_1pm=kappa, speed_mph=mph)
    # keep selection on the moved κ (points may re-sort if κ changed)
    new_pts.sort(key=lambda p: p.kappa_1pm, reverse=True)
    new_idx = min(range(len(new_pts)), key=lambda i: abs(new_pts[i].kappa_1pm - kappa))
    self._set_points(new_pts, selected_idx=new_idx, status="Updated selected point")

  def _refresh_history_list(self) -> None:
    self.history_list.delete(0, tk.END)
    for rec in reversed(self.history[-200:]):
      ts = str(rec.get("created_at", ""))
      note = str(rec.get("note", ""))
      head = str(rec.get("git_head", ""))
      label = f"{ts}  [{head}]  {note}".strip()
      self.history_list.insert(tk.END, label[:120])

  def _on_history_select(self, _ev: tk.Event) -> None:
    # no-op; Load button applies
    pass

  def _selected_history_record(self) -> dict | None:
    sel = self.history_list.curselection()
    if not sel:
      return None
    # list is reversed slice of history
    idx = sel[0]
    slice_hist = list(reversed(self.history[-200:]))
    if idx < 0 or idx >= len(slice_hist):
      return None
    return slice_hist[idx]

  def _load_selected_history(self) -> None:
    rec = self._selected_history_record()
    if rec is None:
      return
    pts = rec.get("points_speed_mph")
    if not isinstance(pts, list) or not pts:
      messagebox.showerror("History load failed", "Selected history entry has no points.")
      return
    new_points: list[ControlPoint] = []
    try:
      for k, mph in pts:
        new_points.append(ControlPoint(kappa_1pm=float(k), speed_mph=float(mph)))
    except Exception:
      messagebox.showerror("History load failed", "Selected history entry points are invalid.")
      return

    # keep within current range and sorted left->right (desc curvature)
    new_points.sort(key=lambda p: p.kappa_1pm, reverse=True)
    new_points = [ControlPoint(kappa_1pm=_clip(p.kappa_1pm, self.kappa_min, self.kappa_max), speed_mph=_clip(p.speed_mph, 0.0, self.y_max_mph)) for p in new_points]
    try:
      self.code_enabled_var.set(bool(rec.get("q_curve_enabled", self.code_enabled_var.get())))
    except Exception:
      pass
    if self.note_var.get().strip() == "":
      try:
        self.note_var.set(str(rec.get("note", "")).strip())
      except Exception:
        pass
    self._set_points(new_points, selected_idx=None, status="Loaded from history")

  def _export_svg_dialog(self) -> None:
    path = filedialog.asksaveasfilename(
      title="Export SVG",
      defaultextension=".svg",
      filetypes=[("SVG", "*.svg")],
      initialfile=f"vtsc_curve_{_git_head_short()}_{_now_iso().replace(':', '')}.svg",
    )
    if not path:
      return
    self._export_svg(path)
    messagebox.showinfo("Exported", f"Wrote {path}")

  def _export_svg(self, path: str) -> None:
    plot = SVGPlot()
    kappas, base_mph = self._curve_samples()
    q_pts = self._q_points()
    tuned_mph = self._tuned_curve_from_q(kappas, base_mph, q_pts)
    plot.write(
      path,
      title=f"VTSC curvature→speed (git { _git_head_short() })",
      kappa_desc=kappas,
      base_speed_mph=base_mph,
      tuned_speed_mph=tuned_mph,
      y_max_mph=float(self.y_max_mph),
      kappa_min=float(self.kappa_min),
      kappa_max=float(self.kappa_max),
      x_scale=str(self.x_scale_var.get()),
    )

  def _apply_to_code(self) -> None:
    q_pts = self._q_points()
    note = self.note_var.get().strip()
    created_at = _now_iso()
    head = _git_head_short()
    enabled = bool(self.code_enabled_var.get())

    # record history before write (in case write fails we still have it)
    rec = {
      "created_at": created_at,
      "git_head": head,
      "note": note,
      "controller_path": os.path.relpath(self.controller_path, _repo_root()),
      "tuning_path": os.path.relpath(self.tuning_path, _repo_root()),
      "kappa_min": float(self.kappa_min),
      "kappa_max": float(self.kappa_max),
      "y_max_mph": float(self.y_max_mph),
      "q_curve_enabled": enabled,
      "points_speed_mph": [[p.kappa_1pm, p.speed_mph] for p in self.points],
      "points_q": [[k, q] for (k, q) in q_pts],
    }
    _append_jsonl(self.history_path, rec)
    self.history.append(rec)
    self._refresh_history_list()

    content_lines = []
    content_lines.append("# This file is auto-generated by tools/vtsc/curve_tuner.py")
    content_lines.append("# Edit via the tuner to preserve history and visualization.")
    content_lines.append("")
    content_lines.append("from __future__ import annotations")
    content_lines.append("")
    content_lines.append(f"Q_CURVE_ENABLED = {enabled}")
    content_lines.append("")
    content_lines.append("# Control points for a multiplicative speed scale q(κ).")
    content_lines.append("# Each point is (curvature_1_per_m, q_multiplier). Interpolated linearly in log10(κ).")
    content_lines.append("Q_CURVE_POINTS: list[tuple[float, float]] = [")
    for k, q in sorted(q_pts, key=lambda kv: kv[0]):
      content_lines.append(f"  ({k:.10g}, {q:.10g}),")
    content_lines.append("]")
    content_lines.append("")
    meta = {
      "created_at": created_at,
      "git_head": head,
      "note": note,
      "source": "tools/vtsc/curve_tuner.py",
    }
    content_lines.append(f"Q_CURVE_META = {json.dumps(meta, sort_keys=True)}")
    content_lines.append("")

    try:
      _write_text(self.tuning_path, "\n".join(content_lines) + "\n")
    except Exception as e:
      messagebox.showerror("Apply failed", f"Failed to write {self.tuning_path}: {e}")
      return

    messagebox.showinfo("Applied", f"Wrote {os.path.relpath(self.tuning_path, _repo_root())}\nAlso appended history.")


def _default_paths() -> tuple[str, str, str]:
  root = _repo_root()
  controller_path = os.path.join(root, "sunnypilot", "selfdrive", "controls", "lib", "vision_turn_controller.py")
  tuning_path = os.path.join(root, "sunnypilot", "selfdrive", "controls", "lib", "vtsc_curve_tuning.py")
  history_path = os.path.join(root, ".cache", "vtsc_curve_tuner", "history.jsonl")
  return controller_path, tuning_path, history_path


def main() -> int:
  parser = argparse.ArgumentParser(description="VTSC curvature→speed curve tuner (Tkinter).")
  controller_path, tuning_path, history_path = _default_paths()
  parser.add_argument("--controller-path", default=controller_path)
  parser.add_argument("--tuning-path", default=tuning_path)
  parser.add_argument("--history-path", default=history_path)
  parser.add_argument("--print-base-mph", type=float, default=None, help="Print baseline mph at κ and exit.")
  parser.add_argument("--export-svg", default=None, help="Headless: write SVG at path and exit.")
  args = parser.parse_args()

  model = CurveModel(args.controller_path)

  if args.print_base_mph is not None:
    mph = model.base_speed_mph(float(args.print_base_mph))
    print(f"κ={args.print_base_mph:.8g}  base={mph:.4f} mph")
    return 0

  if args.export_svg:
    app = type("Tmp", (), {})()
    app.model = model
    app.kappa_max = 2.0e-2
    app.kappa_min = 1.0e-5
    app.y_max_mph = 90.0
    app.n_samples = 250
    kappas = _sample_curve_kappa(app.kappa_max, app.kappa_min, app.n_samples)
    base_mph = [model.base_speed_mph(k) for k in kappas]
    # Use current tuning file if it exists; otherwise q=1.
    q_pts: list[tuple[float, float]] = []
    try:
      mod = {}
      exec(_read_text(args.tuning_path), mod, mod)
      pts = mod.get("Q_CURVE_POINTS", [])
      if isinstance(pts, list) and pts:
        q_pts = [(float(k), float(q)) for (k, q) in pts]
    except Exception:
      q_pts = []
    tuned_mph = [model.tuned_speed_mph(k, q_pts) for k in kappas]
    SVGPlot().write(
      args.export_svg,
      title=f"VTSC curvature→speed (git { _git_head_short() })",
      kappa_desc=kappas,
      base_speed_mph=base_mph,
      tuned_speed_mph=tuned_mph,
      y_max_mph=float(app.y_max_mph),
      kappa_min=float(app.kappa_min),
      kappa_max=float(app.kappa_max),
    )
    print(f"Wrote {args.export_svg}")
    return 0

  root = tk.Tk()
  CurveTunerApp(
    root,
    controller_path=args.controller_path,
    tuning_path=args.tuning_path,
    history_path=args.history_path,
  )
  root.mainloop()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

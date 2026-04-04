#!/usr/bin/env python3
"""
RainViewer "Universal Blue" palette → precipitation-intensity lookup.

RainViewer's radar tiles (color scheme 2 = Universal Blue) encode precipitation
intensity as RGBA pixels. We built this table by sampling live tiles across
9 regions during planning and grouping the 48 distinct alpha=255 palette
entries by color family. dBZ is assigned by category; mm/hour is computed
via the Marshall-Palmer Z-R relation:

    Z = 200 * R^1.6    →    R_mm_per_hr = (10^(dBZ/10) / 200)^(1/1.6)

Families (dBZ ranges grounded in standard WSR-88D radar conventions):
  cyan  (R<150, G>170, B>220) :  5 – 20 dBZ :  0.07 – 0.65 mm/hr : drizzle / light
  blue  (R<30, G 70–165, B>100) : 20 – 35 dBZ :  0.65 – 5.62 mm/hr : light – moderate
  yellow   (R>240, G>170, B<50) : 35 – 40 dBZ :  5.62 – 11.5 mm/hr : heavy
  orange   (R>240, G 80–170, B<50) : 40 – 45 dBZ : 11.5 – 23.7 mm/hr : very heavy
  red      (R 200–255, G<80, B<50) : 45 – 50 dBZ : 23.7 – 48.6 mm/hr : extreme
  dark-red (R 80–199, G<30, B<30)  : 50 – 55 dBZ : 48.6 – 99.9 mm/hr : extreme
  purple   (R>240, G>100, B>200)   : 55+    dBZ : 99.9+  mm/hr    : extreme

Tuning note: these dBZ assignments are field-tunable. If slowdowns trigger at
wrong intensities, adjust the category dBZ values (not the Marshall-Palmer math).
"""

# Only count solid radar pixels; lower-alpha pixels are anti-aliased rendering
# artifacts at the edges of radar cells.
MIN_ALPHA = 200


def _dbz_to_mm_per_hr(dbz: float) -> float:
  """Marshall-Palmer Z-R relation: R_mm_per_hr = (10^(dBZ/10) / 200)^(1/1.6)."""
  if dbz <= 0:
    return 0.0
  z = 10.0 ** (dbz / 10.0)
  return (z / 200.0) ** (1.0 / 1.6)


def _classify_dbz(r: int, g: int, b: int) -> float:
  """Map an RGB tuple to a representative dBZ value for Universal Blue."""
  # Purple / magenta / white — extreme (60 dBZ)
  if r > 240 and b > 200:
    return 60.0

  # Warm colors (yellow → orange → red → dark red)
  if b < 50 and r > 80:
    if r > 240 and g > 170:
      # Yellow: G 180 (bright) → 238 (darker). Darker = higher dBZ within family.
      # Map G in [170, 255] → dBZ [40, 35], more yellow = less intense.
      return 40.0 - (g - 170) / 85.0 * 5.0
    if r > 240 and g > 80:
      # Orange: G 80 → 170 maps to dBZ 45 → 40.
      return 45.0 - (g - 80) / 90.0 * 5.0
    if r >= 200 and g < 80:
      # Red: R 205-255, darker = higher dBZ. Map R in [205, 255] → dBZ [50, 45].
      return 50.0 - (r - 205) / 50.0 * 5.0
    if r < 200 and g < 30:
      # Dark red: R 93-193, darker = higher dBZ. Map R in [93, 193] → dBZ [55, 50].
      return 55.0 - (r - 93) / 100.0 * 5.0
    # Fallthrough warm color
    return 45.0

  # Cyan family (R up to 140, G high, B high) — check BEFORE blue because
  # darker cyan entries have low R but their G>170+B>220 signature distinguishes
  # them from blue family members that lack G>170.
  if r < 150 and g > 170 and b > 220:
    # Within cyan, lighter = lower dBZ. Observed R range 27..136.
    # Map R in [27, 136] → dBZ [20, 5] (brighter cyan = lighter precip).
    # Max 20 stays strictly below blue family minimum (20.5) for monotonicity.
    dbz = 20.0 - (r - 27.0) / (136.0 - 27.0) * 15.0
    return max(5.0, min(20.0, dbz))

  # Blue family (R very low, B dominant, G below cyan threshold)
  if r < 30 and b > 100:
    # Darker blue = higher dBZ in Universal Blue. Use G+B as brightness proxy.
    # Observed blue range: brightness (G+B) from 71+104=175 → 163+224=387.
    # Map G+B in [175, 387] → dBZ [35, 20.5]. Minimum 20.5 so the blue family
    # is strictly more intense than the cyan family at its boundary.
    gb = g + b
    dbz = 35.0 - (gb - 175.0) / (387.0 - 175.0) * 14.5
    return max(20.5, min(35.0, dbz))

  # Unknown opaque pixel — assume light precipitation to fail-safe-ish.
  return 20.0


def rgba_to_mm_per_hr(r: int, g: int, b: int, a: int) -> float:
  """Convert a single RGBA pixel from a RainViewer Universal Blue tile to mm/hr.

  Pixels below MIN_ALPHA (anti-aliased edges, transparency) return 0.0.
  """
  if a < MIN_ALPHA:
    return 0.0
  dbz = _classify_dbz(r, g, b)
  return _dbz_to_mm_per_hr(dbz)


def sample_intensity_max(pixels: bytes, width: int, height: int,
                         center_x: int, center_y: int, window: int = 3) -> float:
  """Return the max mm/hr over a window×window RGBA neighborhood around (cx, cy).

  `pixels` is a flat RGBA byte array (4 bytes per pixel, row-major).
  The window is clamped to the image bounds at edges.
  """
  half = window // 2
  x0 = max(0, center_x - half)
  x1 = min(width - 1, center_x + half)
  y0 = max(0, center_y - half)
  y1 = min(height - 1, center_y + half)

  best = 0.0
  for y in range(y0, y1 + 1):
    row_off = y * width * 4
    for x in range(x0, x1 + 1):
      i = row_off + x * 4
      r, g, b, a = pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]
      mm = rgba_to_mm_per_hr(r, g, b, a)
      if mm > best:
        best = mm
  return best


def severity_from_mm_per_hr(mm_per_hr: float) -> str:
  """Map a mm/hr intensity to a coarse severity string for HUD display."""
  if mm_per_hr < 0.5:
    return "none"
  if mm_per_hr < 2.5:
    return "light"
  if mm_per_hr < 7.5:
    return "moderate"
  return "heavy"

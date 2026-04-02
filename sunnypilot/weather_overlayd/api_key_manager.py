import os

from openpilot.common.params import Params

KEY_PATHS = (
  "/persist/openweather_api_key",
  "/projects/chauffeur/persist/openweather_api_key",
)


def get_api_key() -> str | None:
  params = Params()

  try:
    param_val = params.get("WeatherOverlayManualApiKey")
    if isinstance(param_val, (bytes, bytearray)):
      param_val = param_val.decode("utf-8", errors="ignore")
    if isinstance(param_val, str) and param_val.strip():
      return param_val.strip()
  except Exception:
    pass

  for env_key in ("OPENWEATHER_API_KEY", "WEATHER_OVERLAY_API_KEY"):
    value = os.getenv(env_key)
    if value and value.strip():
      return value.strip()

  for key_path in KEY_PATHS:
    try:
      with open(key_path, encoding="utf-8") as f:
        value = f.readline().strip()
      if value:
        return value
    except OSError:
      continue

  return None

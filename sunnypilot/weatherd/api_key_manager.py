import os

from openpilot.common.params import Params

# Follows the same "canonical filename, two environment-specific locations"
# pattern used by sunnypilot/rtid/api_key_manager.py.
API_KEY_FILENAME = "pirateweather_api_key"
API_KEY_PATHS = (
  f"/persist/{API_KEY_FILENAME}",                          # TICI persistent storage
  f"/projects/chauffeur/persist/{API_KEY_FILENAME}",       # Dev fallback
)


def get_api_key() -> str | None:
  """Resolve the Pirate Weather API key from (in order):

  1. Params: WeatherPointManualApiKey (user override set from the settings UI)
  2. Env: PIRATE_WEATHER_API_KEY
  3. Key files under /persist or /projects/chauffeur/persist
  """
  params = Params()

  try:
    param_val = params.get("WeatherPointManualApiKey")
    if isinstance(param_val, (bytes, bytearray)):
      param_val = param_val.decode("utf-8", errors="ignore")
    if isinstance(param_val, str) and param_val.strip():
      return param_val.strip()
  except Exception:
    pass

  env_val = os.getenv("PIRATE_WEATHER_API_KEY")
  if env_val and env_val.strip():
    return env_val.strip()

  for key_path in API_KEY_PATHS:
    try:
      with open(key_path, encoding="utf-8") as f:
        value = f.readline().strip()
      if value:
        return value
    except OSError:
      continue

  return None

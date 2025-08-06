"""Stub openpilot modules for VTSC testing"""

class MockParams:
    def __init__(self):
        self._params = {
            "VisionTurnSpeedControl": True  # Enable VTSC by default for testing
        }

    def get(self, key, default=None):
        return self._params.get(key, default)

    def get_bool(self, key, default=False):
        return self._params.get(key, default)

    def put(self, key, value):
        self._params[key] = value

class MockConversions:
    MS_TO_MPH = 2.23694
    MPH_TO_MS = 1 / MS_TO_MPH
    KPH_TO_MS = 1 / 3.6
    MS_TO_KPH = 3.6
    DEG_TO_RAD = 3.14159265359 / 180.0
    RAD_TO_DEG = 180.0 / 3.14159265359

class MockModelConstants:
    T_IDXS = list(range(33))

def clip(x, min_val, max_val):
    return max(min_val, min(x, max_val))

# Mock module structure
common = type('common', (), {
    'params': type('params', (), {'Params': MockParams}),
    'conversions': type('conversions', (), {'Conversions': MockConversions}),
    'numpy_fast': type('numpy_fast', (), {'clip': clip}),
})

selfdrive = type('selfdrive', (), {
    'car': type('car', (), {
        'cruise': type('cruise', (), {'V_CRUISE_MAX': 40.0})  # 40 m/s max cruise
    }),
    'modeld': type('modeld', (), {
        'constants': type('constants', (), {'ModelConstants': MockModelConstants})
    }),
    'controls': type('controls', (), {
        'lib': type('lib', (), {
            'drive_helpers': type('drive_helpers', (), {'CONTROL_N': 17})
        })
    })
})

# Create openpilot module structure
openpilot = type('openpilot', (), {
    'common': common,
    'selfdrive': selfdrive
})

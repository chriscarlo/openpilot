#!/usr/bin/env python3
"""
Patch to add debug logging to SLC components
This will help identify where the car speed limit is getting lost
"""

print("=" * 70)
print("DEBUG PATCH FOR SLC CAR SPEED LIMIT")
print("=" * 70)

print("\nTo debug the issue, add these logging statements:\n")

print("1. In /opendbc_repo/opendbc/car/hyundai/carstate.py around line 328:")
print("-" * 60)
print("""
    if "FR_CMR_02_100ms" in cp.vl:
      raw = cp.vl["FR_CMR_02_100ms"].get("ISLW_SpdCluMainDis", 0)
      # DEBUG: Log raw value
      print(f"[HYUNDAI] Dashboard speed limit raw={raw}")
      # 0 = no recognition, 255 = invalid, 253 = unlimited
      if raw not in (0, 255, 253):
        ret_sp.speedLimit = float(raw) * speed_factor
        # DEBUG: Log calculated value
        print(f"[HYUNDAI] Setting ret_sp.speedLimit={ret_sp.speedLimit:.3f} m/s (raw={raw}, factor={speed_factor:.6f})")
      else:
        ret_sp.speedLimit = 0.0
        # DEBUG: Log invalid
        print(f"[HYUNDAI] Invalid speed limit raw={raw}, setting to 0")
""")

print("\n2. In /selfdrive/car/card.py around line 269:")
print("-" * 60)
print("""
    cs_sp_send = messaging.new_message('carStateSP')
    cs_sp_send.valid = CS.canValid
    cs_sp_send.carStateSP = CS_SP
    # DEBUG: Log what we're sending
    print(f"[CARD] Publishing carStateSP.speedLimit={CS_SP.speedLimit:.3f} m/s")
    self.pm.send('carStateSP', cs_sp_send)
""")

print("\n3. In /sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_resolver.py line 56:")
print("-" * 60)
print("""
  def _get_from_car_state(self, sm: messaging.SubMaster) -> None:
    self._reset_limit_sources(Source.car_state)
    car_speed_limit = sm['carStateSP'].speedLimit
    # DEBUG: Log what we received
    print(f"[RESOLVER] Got carStateSP.speedLimit={car_speed_limit:.3f} m/s")
    self._limit_solutions[Source.car_state] = car_speed_limit
    self._distance_solutions[Source.car_state] = 0.
""")

print("\n4. In /sunnypilot/selfdrive/controls/lib/speed_limit_controller/speed_limit_controller.py around line 355:")
print("-" * 60)
print("""
    self._speed_limit, self._distance, self._source = self._resolver.resolve(v_ego, self.speed_limit, sm)
    # DEBUG: Log resolution result
    print(f"[SLC] Resolved: limit={self._speed_limit:.3f} m/s, source={self._source}, policy={self._policy}")
""")

print("\n" + "=" * 70)
print("After adding these debug statements, run openpilot and check the logs.")
print("This will show exactly where the speed limit data is getting lost.")
print("=" * 70)
# RTI Pipeline

## Ownership Map

- Ingress and API coordination:
  `sunnypilot/rtid/api_key_manager.py`, `sunnypilot/rtid/waze_api_client.py`, `sunnypilot/rtid/rtid.py`
- Threat processing and same-road logic:
  `sunnypilot/rtid/threat_detector.py`, `sunnypilot/rtid/street_name_matcher.py`
- Planner slowdown path:
  `sunnypilot/selfdrive/controls/lib/rti_controller.py`, `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- Process and message wiring:
  `system/manager/process_config.py`, `selfdrive/controls/plannerd.py`, `cereal/custom.capnp`, `common/params_keys.h`
- Onroad UI:
  `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`, `selfdrive/ui/sunnypilot/ui.cc`
- Offroad UI:
  `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.cc`, `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.cc`

## Branch Facts Worth Rechecking Before Tuning

- `rtid` fetches on a 30 second interval and reuses cached traffic data up to 300 seconds old.
- `rtiStateSP` is published at 1 Hz and consumed by both the planner path and the RTI HUD.
- API key load priority in `RTIDaemon` is:
  `RTIManualApiKey` param first, then env or persist lookup through `api_key_manager.py`.
- `RTISettingsPanel::loadWazeApiKey()` seeds `RTIManualApiKey` from env or persist paths when the panel is constructed.
- `ThreatDetector` publishes at most five processed threats, sorted nearest-first.
- `RTIController` re-checks `RTIEnabled` live, but most RTI distance and speed tuning knobs are still init-cached across RTID, `ThreatDetector`, and `RTIController`.
- Current RTI settings UI shows miles for range knobs and mph for custom speed reduction while storing meter and km/h params underneath.
- `RTIDataSource` and `RTIAggressiveness` have a lot of legacy surface area in params, UI, and tests; confirm runtime effect in the current branch before tuning around them.

## Test Map

- RTID daemon behavior:
  `sunnypilot/rtid/tests/test_rtid.py`
- Threat processing, same-road logic, and recommendation engine:
  `sunnypilot/rtid/tests/test_threat_detector.py`
- Schema contract and enum coverage:
  `sunnypilot/rtid/tests/test_schema_synchronization.py`
- End-to-end RTID integration:
  `sunnypilot/rtid/tests/test_integration_flow.py`
- Planner-side RTI rampdown and filter behavior:
  `sunnypilot/selfdrive/controls/lib/tests/rti/test_rti_rampdown_unit.py`
- Broad RTI integration smoke:
  `selfdrive/test/test_rti_integration.py`
- Historical live-simulation harness:
  `selfdrive/test/test_rti_live_validation.py`
- UI-specific RTI dropdown regression coverage:
  `selfdrive/ui/tests/test_rti_dropdowns.cc`

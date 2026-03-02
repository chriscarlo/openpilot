from types import SimpleNamespace

from openpilot.selfdrive.locationd.paramsd import has_fresh_valid_carstate


def make_sm(*, seen: bool, valid: bool, frame: int, recv_frame: int):
  return SimpleNamespace(
    seen={"carState": seen},
    valid={"carState": valid},
    frame=frame,
    recv_frame={"carState": recv_frame},
  )


def test_has_fresh_valid_carstate_accepts_recent_valid_sample():
  sm = make_sm(seen=True, valid=True, frame=200, recv_frame=198)
  assert has_fresh_valid_carstate(sm, max_age_frames=5)


def test_has_fresh_valid_carstate_rejects_stale_sample():
  sm = make_sm(seen=True, valid=True, frame=200, recv_frame=180)
  assert not has_fresh_valid_carstate(sm, max_age_frames=5)


def test_has_fresh_valid_carstate_rejects_unseen_or_invalid():
  sm_unseen = make_sm(seen=False, valid=True, frame=200, recv_frame=199)
  sm_invalid = make_sm(seen=True, valid=False, frame=200, recv_frame=199)
  assert not has_fresh_valid_carstate(sm_unseen, max_age_frames=5)
  assert not has_fresh_valid_carstate(sm_invalid, max_age_frames=5)

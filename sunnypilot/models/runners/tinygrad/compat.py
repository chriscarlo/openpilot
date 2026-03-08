def get_captured_input_info(captured) -> list[tuple]:
  if hasattr(captured, "expected_input_info"):
    return captured.expected_input_info
  if hasattr(captured, "expected_st_vars_dtype_device"):
    return captured.expected_st_vars_dtype_device
  raise AttributeError("CapturedJit missing expected input metadata")

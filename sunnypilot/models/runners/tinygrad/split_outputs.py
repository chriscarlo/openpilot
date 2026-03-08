import numpy as np


NumpyDict = dict[str, np.ndarray]


def merge_split_model_outputs(policy_output: NumpyDict, vision_output: NumpyDict,
                              off_policy_output: NumpyDict | None = None) -> NumpyDict:
  outputs = {**policy_output, **vision_output}
  if off_policy_output is not None:
    outputs.update(off_policy_output)
  return outputs

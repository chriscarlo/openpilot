import os
from openpilot.system.hardware import TICI

# In many PC/dev test environments we don't have the compiled model runner extensions
# available. Avoid failing import-time so non-model unit tests (e.g., VTSC) can run.
try:
  from openpilot.sunnypilot.modeld.runners.runmodel_pyx import RunModel, Runtime  # type: ignore
except ModuleNotFoundError:
  RunModel = None  # type: ignore
  Runtime = None  # type: ignore

USE_THNEED = int(os.getenv('USE_THNEED', str(int(TICI))))
USE_SNPE = int(os.getenv('USE_SNPE', str(int(TICI))))

if RunModel is None:
  class ModelRunner:
    THNEED = 'THNEED'
    SNPE = 'SNPE'
    ONNX = 'ONNX'

    def __new__(cls, *args, **kwargs):
      raise ModuleNotFoundError(
        "Sunnypilot model runner extensions are not available (runmodel_pyx missing). "
        "Build the model runner or run on a supported device environment."
      )
else:
  class ModelRunner(RunModel):
    THNEED = 'THNEED'
    SNPE = 'SNPE'
    ONNX = 'ONNX'

    def __new__(cls, paths, *args, **kwargs):
      if ModelRunner.THNEED in paths and USE_THNEED:
        from openpilot.sunnypilot.modeld.runners.thneedmodel_pyx import ThneedModel as Runner
        runner_type = ModelRunner.THNEED
      elif ModelRunner.SNPE in paths and USE_SNPE:
        from openpilot.sunnypilot.modeld.runners.snpemodel_pyx import SNPEModel as Runner
        runner_type = ModelRunner.SNPE
      elif ModelRunner.ONNX in paths:
        from openpilot.sunnypilot.modeld.runners.onnxmodel import ONNXModel as Runner
        runner_type = ModelRunner.ONNX
      else:
        raise Exception("Couldn't select a model runner, make sure to pass at least one valid model path")

      return Runner(str(paths[runner_type]), *args, **kwargs)

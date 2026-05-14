import os
import shutil
import uuid


from openpilot.common.params import Params
from openpilot.system.hardware import PC
from openpilot.system.hardware.hw import Paths
from openpilot.system.hardware.hw import DEFAULT_DOWNLOAD_CACHE_ROOT

class OpenpilotPrefix:
  def __init__(self, prefix: str = None, create_dirs_on_enter: bool = True, clean_dirs_on_exit: bool = True, shared_download_cache: bool = False):
    self.prefix = prefix if prefix else str(uuid.uuid4().hex[0:15])
    self.msgq_path = os.path.join(Paths.shm_path(), self.prefix)
    self.create_dirs_on_enter = create_dirs_on_enter
    self.clean_dirs_on_exit = clean_dirs_on_exit
    self.shared_download_cache = shared_download_cache

  def __enter__(self):
    self.original_prefix = os.environ.get('OPENPILOT_PREFIX', None)
    os.environ['OPENPILOT_PREFIX'] = self.prefix

    if self.create_dirs_on_enter:
      self.create_dirs()

    if self.shared_download_cache:
      os.environ["COMMA_CACHE"] = DEFAULT_DOWNLOAD_CACHE_ROOT

    return self

  def __exit__(self, exc_type, exc_obj, exc_tb):
    if self.clean_dirs_on_exit:
      self.clean_dirs()
    try:
      del os.environ['OPENPILOT_PREFIX']
      if self.original_prefix is not None:
        os.environ['OPENPILOT_PREFIX'] = self.original_prefix
    except KeyError:
      pass
    return False

  def create_dirs(self):
    os.makedirs(self.msgq_path, exist_ok=True)
    os.makedirs(Paths.log_root(), exist_ok=True)

  def clean_dirs(self):
    params_path = Params().get_param_path()
    if os.path.exists(params_path):
      if os.path.islink(params_path):
        shutil.rmtree(os.path.realpath(params_path), ignore_errors=True)
        os.remove(params_path)
      elif os.path.isdir(params_path):
        shutil.rmtree(params_path, ignore_errors=True)
      else:
        os.remove(params_path)
    shutil.rmtree(self.msgq_path, ignore_errors=True)
    if PC:
      shutil.rmtree(Paths.log_root(), ignore_errors=True)
    if not os.environ.get("COMMA_CACHE", False):
      shutil.rmtree(Paths.download_cache_root(), ignore_errors=True)
    shutil.rmtree(Paths.comma_home(), ignore_errors=True)

import sys
import threading
import time
from collections import defaultdict, deque


if sys.platform != "win32":
  raise ImportError("msgq.ipc_pyx must be built with scons outside Windows")


class IpcError(Exception):
  pass


class MultiplePublishersError(IpcError):
  pass


_SUBSCRIBERS = defaultdict(list)
_LOCK = threading.RLock()
_FAKE_PREFIX = b""
_FAKE_EVENTS_ENABLED = False


def _decode_endpoint(endpoint):
  return endpoint.decode("utf-8") if isinstance(endpoint, bytes) else str(endpoint)


def toggle_fake_events(enabled):
  global _FAKE_EVENTS_ENABLED
  _FAKE_EVENTS_ENABLED = bool(enabled)


def set_fake_prefix(prefix):
  global _FAKE_PREFIX
  _FAKE_PREFIX = prefix.encode("utf-8") if isinstance(prefix, str) else bytes(prefix)


def get_fake_prefix():
  return _FAKE_PREFIX


def delete_fake_prefix():
  set_fake_prefix(b"")


def wait_for_one_event(events, timeout=-1):
  deadline = None if timeout < 0 else time.monotonic() + timeout / 1000.0
  while True:
    for event in events:
      if event.peek():
        return event
    if deadline is not None and time.monotonic() >= deadline:
      return None
    time.sleep(0.001)


class Event:
  def __init__(self):
    self._event = threading.Event()

  def set(self):
    self._event.set()

  def clear(self):
    self._event.clear()

  def wait(self, timeout=-1):
    timeout_s = None if timeout < 0 else timeout / 1000.0
    self._event.wait(timeout_s)

  def peek(self):
    return self._event.is_set()

  @property
  def fd(self):
    return -1

  @property
  def ptr(self):
    return id(self)


class SocketEventHandle:
  _handles = {}

  def __init__(self, endpoint, identifier, override):
    identifier_bytes = identifier.encode("utf-8") if isinstance(identifier, str) else bytes(identifier)
    key = (_decode_endpoint(endpoint), identifier_bytes, bool(override))
    self._enabled = False
    self.recv_called_event = Event()
    self.recv_ready_event = Event()
    SocketEventHandle._handles[key] = self

  @property
  def enabled(self):
    return self._enabled

  @enabled.setter
  def enabled(self, value):
    self._enabled = bool(value)


class Context:
  def term(self):
    pass


class Poller:
  def __init__(self):
    self.sub_sockets = []

  def registerSocket(self, socket):
    self.sub_sockets.append(socket)

  def poll(self, timeout):
    deadline = time.monotonic() + max(0, timeout) / 1000.0
    while True:
      ready = [socket for socket in self.sub_sockets if socket._has_messages()]
      if ready or timeout == 0 or time.monotonic() >= deadline:
        return ready
      time.sleep(0.001)


class SubSocket:
  def __init__(self):
    self.endpoint = None
    self.conflate = False
    self.timeout = None
    self._queue = deque()

  def connect(self, context, endpoint, address=b"127.0.0.1", conflate=False):
    self.endpoint = _decode_endpoint(endpoint)
    self.conflate = bool(conflate)
    with _LOCK:
      _SUBSCRIBERS[self.endpoint].append(self)

  def setTimeout(self, timeout):
    self.timeout = timeout

  def _has_messages(self):
    with _LOCK:
      return bool(self._queue)

  def _push(self, data):
    with _LOCK:
      if self.conflate:
        self._queue.clear()
      self._queue.append(bytes(data))

  def receive(self, non_blocking=False):
    timeout = 0 if non_blocking else self.timeout
    deadline = None if timeout is None else time.monotonic() + timeout / 1000.0
    while True:
      with _LOCK:
        if self._queue:
          return self._queue.popleft()
      if non_blocking or (deadline is not None and time.monotonic() >= deadline):
        return None
      time.sleep(0.001)


class PubSocket:
  def __init__(self):
    self.endpoint = None

  def connect(self, context, endpoint):
    self.endpoint = _decode_endpoint(endpoint)

  def send(self, data):
    with _LOCK:
      subscribers = list(_SUBSCRIBERS.get(self.endpoint, []))
    for subscriber in subscribers:
      subscriber._push(data)
    return len(data)

  def all_readers_updated(self):
    return True

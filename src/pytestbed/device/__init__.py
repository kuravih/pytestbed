import ast
from datetime import datetime
from enum import Enum, auto

import numpy as np
import toml
import zmq
from pykato.log import setup_logger
from pyshmio import SharedMemory

device_logger = setup_logger("Device", terminator="\n")


class Device:
    """Device.

    Attributes:
        name: str
            name
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Name must be a string")
        self._name = value

    @property
    def preview_window_id(self) -> str:
        return f"{self.name}_preview_window"

    @property
    def info_window_id(self) -> str:
        return f"{self.name}_info_window"

    @property
    def settings_window_id(self) -> str:
        return f"{self.name}_settings_window"

    @property
    def sampling_worker_id(self) -> str:
        return f"{self.name}_sampling_worker"

    @property
    def storage_worker_id(self) -> str:
        return f"{self.name}_storage_worker"

    def __del__(self):
        device_logger.info("Device object %s removed", self.name)


class Stream(SharedMemory):
    """Stream."""

    class Kind(Enum):
        CAMERA = auto()
        SLM = auto()
        DM = auto()

        @classmethod
        def from_str(cls, kind: str):
            if kind.upper() == "CAMERA":
                return cls.CAMERA
            elif kind.upper() == "DM":
                return cls.DM
            elif kind.upper() == "SLM":
                return cls.SLM
            else:
                raise ValueError(f"Unknown kind: {kind}")

        def to_str(self):
            return self.name.upper()

    __slots__ = ("_kind", "_sn", "_pxmax", "_full_shape", "_port", "_shape")

    def __init__(self, source: str | SharedMemory):
        """Construct stream object.

        Parameters:
            source: str | SharedMemory
                Stream name or existing SharedMemory to attach to
        """
        if isinstance(source, str):
            super().__init__(source)
        else:
            super().__init__(source.name)

        self.kind = self.keywords["KIND"].value
        self._sn = self.keywords["SN"].value
        self._pxmax = self.keywords["PXMAX"].value
        self._full_shape = (self.keywords["FULL.W"].value, self.keywords["FULL.H"].value)
        self._port = self.keywords["PORT"].value

    @property
    def kind(self) -> Kind:
        return self._kind

    @kind.setter
    def kind(self, value: Kind | str):
        if isinstance(value, self.Kind):
            self._kind = value
        elif isinstance(value, str):
            self._kind = self.Kind.from_str(value)
        else:
            raise TypeError(f"kind must be Kind or str, got {type(value).__name__}")

    @property
    def sn(self) -> str:
        return self._sn

    @property
    def pxmax(self) -> float | int:
        return self._pxmax

    @property
    def full_shape(self) -> tuple[int, int]:
        return self._full_shape

    @property
    def port(self) -> int:
        return self._port

    @property
    def dtype(self):
        return self.ndarray.dtype

    def get_data(self) -> np.ndarray:
        self.post_request()
        self.wait_for_response()
        return self.ndarray

    def set_data(self, array: np.ndarray):
        self.wait_for_request()
        self.ndarray[:] = array.ravel()
        self.post_response()


class DeviceStream(Device):
    """StreamDevice to wrap a Stream."""

    __slots__ = ("_stream",)

    def __init__(self, stream: Stream):
        super().__init__(stream.name)
        self._stream = stream

    @property
    def stream(self) -> Stream:
        return self._stream

    @property
    def kind(self) -> Stream.Kind:
        return self._stream.kind

    @property
    def sn(self) -> str:
        return self._stream.sn

    @property
    def pxmax(self) -> float | int:
        return self._stream.pxmax

    @property
    def full_shape(self) -> tuple[int, int]:
        return self._stream.full_shape

    @property
    def port(self) -> int:
        return self._stream.port

    @property
    def creation_time(self) -> datetime:
        return self._stream.creation_time

    @property
    def last_access_time(self) -> datetime:
        return self._stream.last_access_time

    def update_keywords(self):
        self._stream.update_keywords()


class ZMQLink:
    """ZMQ link."""

    __slots__ = ("_uri", "_context", "_socket", "_connected")

    def __init__(self, uri: str | None = None, address: str | None = None, port: int | None = None):
        super().__init__()
        if (uri is None) and (address is None) and (port is None):
            self._uri = "tcp://127.0.0.1:555"
        elif (uri is None) and (address is None) and (port is not None):
            self._uri = f"tcp://127.0.0.1:{port}"
        elif (uri is None) and (address is not None) and (port is None):
            self._uri = f"tcp://{address}:5555"
        elif (uri is None) and (address is not None) and (port is not None):
            self._uri = f"tcp://{address}:{port}"
        elif uri is not None:
            self._uri = uri
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._connected = False

    @property
    def uri(self) -> str:
        return self._uri

    def connect(self):
        self._socket.connect(self._uri)
        self._connected = True

    def disconnect(self):
        self._socket.disconnect(self._uri)
        self._connected = False

    def send(self, message: str):
        self._socket.send_string(message)

    def receive(self) -> str:
        return self._socket.recv_string()

    def send_command(self, command: dict) -> dict:
        self.send(toml.dumps(command))
        reply = toml.loads(self.receive())
        if "settings" in reply:
            if "roi" in reply["settings"]:
                reply["settings"]["roi"] = ast.literal_eval(reply["settings"]["roi"])
        return reply

    def sync_settings(self):
        command = {"settings": "sync"}
        reply = self.send_command(command)
        return reply["settings"]

    def close(self):
        self._socket.close()
        self._context.term()

    def __del__(self):
        self.close()

    def is_connected(self) -> bool:
        return self._connected

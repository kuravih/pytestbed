import numpy as np
from pykato.log import setup_logger

from pytestbed import SinkSample
from pytestbed.function import is_modulator_calibration_file_valid, read_modulator_calibration_file

from ..function import command_to_deflection, deflection_to_command
from . import DeviceStream, Stream, ZMQLink

logger = setup_logger("Modulator", terminator="\n")


class ModulatorStream(DeviceStream):
    __slots__ = ("_shape", "_blank", "_max_radius", "_shape", "_sample", "_calibration_file", "_calibration", "post_request", "wait_for_response")

    def __init__(self, stream: Stream):
        super().__init__(stream)
        self._max_radius = self.stream.keywords["RADMAX"].value
        self._shape = (2 * self._max_radius, 2 * self._max_radius)
        self._blank = np.zeros(self._shape)
        self._sample = SinkSample(self.last_access_time, self.frame_rate_fps, self.center, self.radius, self.blank)
        self._calibration_file: str | None = None
        self._calibration: dict[str, np.ndarray] | None = None
        self.post_request = self.stream.post_request
        self.wait_for_response = self.stream.wait_for_response

    @property
    def max_radius(self) -> int:
        return self._max_radius

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def blank(self) -> np.ndarray:
        return self._blank

    @property
    def sample(self) -> SinkSample:
        return self._sample

    @property
    def frame_rate_fps(self) -> float:
        return self.stream.keywords["FRMRATE"].value

    @property
    def center(self) -> tuple[float, float]:
        return (self.stream.keywords["CENTER.X"].value, self.stream.keywords["CENTER.Y"].value)

    @property
    def radius(self) -> float:
        return self.stream.keywords["RADIUS"].value

    @property
    def calibration(self) -> dict[str, np.ndarray] | None:
        return self._calibration

    @property
    def calibration_file(self) -> str | None:
        return self._calibration_file

    @calibration_file.setter
    def calibration_file(self, value: str | None):
        if value is not None and is_modulator_calibration_file_valid(value, self.shape):
            self._calibration_file = value
            self._calibration = read_modulator_calibration_file(value)
        else:
            self._calibration_file = None
            self._calibration = None

    @property
    def presets_window_id(self) -> str:
        return f"{self.name}_presets_window"

    @property
    def vlim(self) -> tuple[float, float] | tuple[int, int]:
        _vlim = (0, self.pxmax)
        if self.calibration is not None:
            return (np.min(command_to_deflection(_vlim[0], self.calibration["slope"], self.calibration["flat"])), np.max(command_to_deflection(_vlim[1], self.calibration["slope"], self.calibration["flat"])))
        return _vlim

    @property
    def vrange(self) -> float | int:
        _vlim = self.vlim
        return _vlim[1] - _vlim[0]

    def push_command(self, command: np.ndarray) -> SinkSample:
        command_float = command.astype(float)
        if self.calibration is not None:
            command_adu_float = deflection_to_command(command_float, self.calibration["slope"], self.calibration["flat"])
        else:
            command_adu_float = command_float
        command_adu_uint16 = np.clip(command_adu_float, 0, self.pxmax).astype(np.uint16)
        self.stream.set_data(command_adu_uint16)
        self._sample = SinkSample(self.last_access_time, self.frame_rate_fps, self.center, self.radius, command_float.copy())
        return self._sample


class Modulator(ModulatorStream):
    """Modulator"""

    __slots__ = ("_link", "_settings")

    def __init__(self, stream: Stream):
        super().__init__(stream)
        self._link: ZMQLink | None = None
        self._settings: dict[str, float | dict[str, float]] | None = None
        if self.stream.port != -1:
            self._link = ZMQLink(port=self.stream.port)
            self._link.connect()
            self.sync_settings()

    @property
    def link(self) -> ZMQLink | None:
        return self._link

    @property
    def settings(self) -> dict[str, float | dict[str, float]] | None:
        return self._settings

    def sync_settings(self):
        self._settings = self.link.sync_settings()
        logger.info("self._settings : %s", self._settings)

    def move_center(self, x: int, y: int):
        """Move roi by x y amount.

        Parameters:
            x: float
                move horizontally
            y: float
                move vertically
        """
        command = {"settings": {"nudge": {"x": x, "y": y}}}
        reply = self.link.send_command(command)
        self._settings["center"] = reply["settings"]["center"]
        logger.info("self._settings : %s", self._settings)

    def set_radius(self, radius: float):
        """Change radius.

        Parameter:
            radius: float
                radius
        """
        command = {"settings": {"radius": radius}}
        reply = self.link.send_command(command)
        self._settings["radius"] = reply["settings"]["radius"]
        logger.info("self._settings : %s", self._settings)

    def __del__(self):
        logger.info("Modulator object %s removed", self.name)
        super().__del__()

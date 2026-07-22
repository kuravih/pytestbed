import numpy as np
from pykato.log import setup_logger

from pytestbed import SourceSample

from . import DeviceStream, Stream, ZMQLink

logger = setup_logger("Camera", terminator="\n")


class CameraStream(DeviceStream):
    __slots__ = ("_shape", "_blank", "_sample", "wait_for_request", "post_response")

    def __init__(self, stream: Stream):
        super().__init__(stream)
        self._shape = (self.stream.keywords["HEIGHT"].value, self.stream.keywords["WIDTH"].value)
        self._blank = np.zeros(self._shape)
        self._sample = SourceSample(self.last_access_time, self.exposure_time_s, self.gain, self.frame_rate_fps, self.temperature_c, self.roi, self.blank)
        self.wait_for_request = self.stream.wait_for_request
        self.post_response = self.stream.post_response

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def blank(self) -> np.ndarray:
        return self._blank

    @property
    def sample(self) -> SourceSample:
        return self._sample

    @property
    def exposure_time_s(self) -> int:
        return self.stream.keywords["EXPTIME"].value

    @property
    def gain(self) -> float:
        return self.stream.keywords["GAIN"].value

    @property
    def frame_rate_fps(self) -> float:
        return self.stream.keywords["FRMRATE"].value

    @property
    def temperature_c(self) -> float:
        return self.stream.keywords["TEMP"].value

    @property
    def roi(self) -> dict[str, tuple[int, int]]:
        return {"br": (self.stream.keywords["ROI.BR.X"].value, self.stream.keywords["ROI.BR.Y"].value), "tl": (self.stream.keywords["ROI.TL.X"].value, self.stream.keywords["ROI.TL.Y"].value)}

    def pull_capture(self) -> SourceSample:
        capture = self.stream.get_data().reshape(self.shape)
        self._sample = SourceSample(self.last_access_time, self.exposure_time_s, self.gain, self.frame_rate_fps, self.temperature_c, self.roi, capture.copy())
        return self._sample


class Camera(CameraStream):
    __slots__ = ("_link", "_settings")

    def __init__(self, stream: Stream):
        super().__init__(stream)
        self._link: ZMQLink | None = None
        self._settings: dict[str, tuple[tuple[int, int], tuple[int, int]] | float] | None = None
        if self.stream.port != -1:
            self._link = ZMQLink(port=self.stream.port)
            self._link.connect()
            self.sync_settings()

    @property
    def link(self) -> ZMQLink | None:
        return self._link

    @property
    def settings(self) -> dict[str, tuple[tuple[int, int], tuple[int, int]] | float] | None:
        return self._settings

    def sync_settings(self):
        self._settings = self.link.sync_settings()
        logger.info("self._settings : %s", self._settings)

    def set_exposure_time_s(self, exposure_time_s: float):
        """Set the exposure time of the camera.

        Parameter:
            exposure_time_s: float
                Exposure time in s
        """
        command = {"settings": {"exposureTime_s": float(exposure_time_s)}}
        reply = self.link.send_command(command)
        self._settings["exposureTime_s"] = reply["settings"]["exposureTime_s"]
        logger.info("self._settings : %s", self._settings)

    def set_gain(self, gain: float):
        """Set the gain of the camera.

        Parameter:
            gain: int
                gain
        """
        command = {"settings": {"gain": float(gain)}}
        reply = self.link.send_command(command)
        self._settings["gain"] = reply["settings"]["gain"]
        logger.info("self._settings : %s", self._settings)

    def set_temperature_c(self, temperature_c: float):
        """Set the temperature of the camera.

        Parameter:
            temperature_c: float
                temperature in c
        """
        command = {"settings": {"temperature_C": float(temperature_c)}}
        reply = self.link.send_command(command)
        self._settings["temperature_C"] = reply["settings"]["temperature_C"]
        logger.info("self._settings : %s", self._settings)

    def move_roi(self, x: int, y: int):
        """Move roi by x y amount.

        Parameters:
            x: float
                move horizontally
            y: float
                move vertically
        """
        command = {"settings": {"nudge": {"x": int(x), "y": int(y)}}}
        reply = self.link.send_command(command)
        self._settings["roi"] = reply["settings"]["roi"]
        logger.info("self._settings : %s", self._settings)

    def __del__(self):
        logger.info("Camera object %s removed", self.name)
        super().__del__()

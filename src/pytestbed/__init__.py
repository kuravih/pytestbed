from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum, auto

import numpy as np

DTYPE_MAP = {
    np.uint8: 1,
    np.int8: 2,
    np.uint16: 3,
    np.int16: 4,
    np.uint32: 5,
    np.int32: 6,
    np.uint64: 7,
    np.int64: 8,
    np.float32: 9,
    np.float64: 10,
}
INV_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}

HEADER_FORMAT = "=7s2HB"  # 7 char tag + unsigned short width + unsigned short height + unsigned byte datatype

SRC_TAG = b"SRCSMPL"
SRC_HEADER_FORMAT = "=5d4H"  # double timestamp + double frame_rate_fps + double temperature_c + double gain + double exposure_time_s + unsigned short roi.tl.x + unsigned short roi.tl.y + unsigned short roi.br.x + unsigned short roi.br.y


@dataclass(slots=True)
class SourceSample:
    last_access_time: datetime
    exposure_time_s: float
    gain: float
    frame_rate_fps: float
    temperature_c: float
    roi: dict[str, tuple[int, int]]
    capture: np.ndarray


SNK_TAG = b"SNKSMPL"
SNK_HEADER_FORMAT = "=2d3H"  # double timestamp + double frame_rate_fps + unsigned short radius + unsigned short center.x + unsigned short center.y


@dataclass(slots=True)
class SinkSample:
    last_access_time: datetime
    frame_rate_fps: float
    center: tuple[float, float]
    radius: float
    command: np.ndarray


class Flip(Enum):
    NEG = auto()
    POS = auto()

    @classmethod
    def from_bool(cls, value: bool) -> "Flip":
        """Construct a Flip from a boolean value.

        Parameters:
            value: bool
                True maps to POS, False maps to NEG.

        Returns: Flip
            The corresponding Flip enum value.
        """
        return cls.POS if value else cls.NEG

    def to_bool(self) -> bool:
        """Convert the Flip to a boolean.

        Returns: bool
            True if POS, False if NEG.
        """
        return self is Flip.POS


class Rotation(Enum):
    UP = auto()
    LEFT = auto()
    DOWN = auto()
    RIGHT = auto()

    @classmethod
    def from_int(cls, i: int) -> "Rotation":
        """Construct a Rotation from an integer.

        Parameters:
            i: int
                Rotation index: 0=UP, 1=LEFT, 2=DOWN, 3=RIGHT.

        Returns: Rotation
            The corresponding Rotation enum value.
        """
        return {0: cls.UP, 1: cls.LEFT, 2: cls.DOWN, 3: cls.RIGHT}[i]

    def to_int(self):
        """Convert the Rotation to an integer.

        Returns: int
            0 for UP, 1 for LEFT, 2 for DOWN, 3 for RIGHT.
        """
        return {Rotation.UP: 0, Rotation.LEFT: 1, Rotation.DOWN: 2, Rotation.RIGHT: 3}[self]


class PairwiseProbeDirection(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()

    def to_str(self) -> str:
        """Return the lowercase string name of the direction.

        Returns: str
            "horizontal" or "vertical".
        """
        return self.name.lower()


class DOTFProbeDirection(IntEnum):
    RIGHT = 3
    BOTTOM = 6
    LEFT = 9
    TOP = 12

    def to_str(self) -> str:
        """Return the zero-padded string representation of the direction value.

        Returns: str
            Two-character zero-padded integer string, e.g. "03", "06", "09", "12".
        """
        return f"{self.value:02d}"

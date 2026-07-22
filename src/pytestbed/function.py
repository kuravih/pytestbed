import struct
from datetime import datetime
from io import FileIO

import cloudpickle
import numpy as np
from astropy.io import fits
from hcipy.util import large_poisson
from numpy.typing import NDArray
from pykato.function import airy_fn, box, generate_coordinates, invert_2x2_arrays, least_squares_fit_2d
from pykato.log import setup_logger
from pyshmio import DataType, Keyword, KeywordType, SharedMemory
from scipy import ndimage
from scipy.interpolate import CubicSpline
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk

from . import DTYPE_MAP, HEADER_FORMAT, INV_DTYPE_MAP, SNK_HEADER_FORMAT, SNK_TAG, SRC_HEADER_FORMAT, SRC_TAG, DOTFProbeDirection, Flip, PairwiseProbeDirection, Rotation, SinkSample, SourceSample

logger = setup_logger(name="function", terminator="\n")


def is_camera_calibration_file_valid(filename: str, shape: tuple[int, int]) -> bool:
    """Return True if the camera calibration FITS file contains 3 image frames, a QE table, full well capacity, and bit depth, all matching the expected shape.

    Parameters:
        filename: str
            Path to the camera calibration FITS file.
        shape: tuple[int, int]
            Expected (width, height) of each calibration frame.

    Returns: bool
        True if valid, False otherwise.
    """
    with fits.open(filename) as hdul:
        if len(hdul[0].data) != 3:
            return False
        if not ((hdul[0].data[0].shape == shape) & (hdul[0].data[1].shape == shape) & (hdul[0].data[2].shape == shape)):
            return False
        if "QE" not in hdul:
            logger.info('"QE" not in hdul')
            return False
        qe_names = hdul["QE"].data.names
        if "wavelength_nm" not in qe_names or "quantum_efficiency" not in qe_names:
            logger.info('"wavelength_nm" not in qe_names or "quantum_efficiency" not in qe_names')
            return False
        if "FULLWELL" not in hdul[0].header or "BITDEPTH" not in hdul[0].header or "GAIN" not in hdul[0].header:
            logger.info('"FULLWELL" not in hdul[0].header or "BITDEPTH" not in hdul[0].header')
            return False
        return True


def read_camera_calibration_file(filename: str, roi: dict[str, tuple[int, int]] | None = None) -> dict[str, np.ndarray | float | int]:
    """Load dark_rate, bias, read_noise, quantum_efficiency, full_well_capacity, and bit_depth from a camera calibration FITS file.

    Parameters:
        filename: str
            Path to the camera calibration FITS file.

    Returns: dict
        Dict with keys:
            dark_rate: np.ndarray
                per-pixel dark current rate (ADU/s).
            bias: np.ndarray
                per-pixel bias frame (ADU).
            read_noise: np.ndarray
                per-pixel read noise (ADU).
            quantum_efficiency: np.
                shape (2, N): row 0 = wavelength (nm), row 1 = QE fraction.
            full_well_capacity: float
                full well capacity (electrons).
            bit_depth: int
                ADC bit depth.
    """
    with fits.open(filename) as hdul:
        dark_rate_data = hdul[0].data[0]
        bias_data = hdul[0].data[1]
        read_noise_data = hdul[0].data[2]
        gain = hdul[0].header["GAIN"]
        full_well_capacity = hdul[0].header["FULLWELL"]
        bit_depth = hdul[0].header["BITDEPTH"]
        qe_data = hdul["QE"].data
        λ_m_qe_perc_data = np.stack([qe_data["wavelength_nm"], qe_data["quantum_efficiency"]])
        if roi is None:
            roi = {"br": (None, None), "tl": (None, None)}
        return {"dark_rate": dark_rate_data[roi["tl"][1] : roi["br"][1], roi["tl"][0] : roi["br"][0]], "bias": bias_data[roi["tl"][1] : roi["br"][1], roi["tl"][0] : roi["br"][0]], "read_noise": read_noise_data[roi["tl"][1] : roi["br"][1], roi["tl"][0] : roi["br"][0]], "quantum_efficiency": λ_m_qe_perc_data, "gain": gain, "full_well_capacity": full_well_capacity, "bit_depth": bit_depth}


def write_camera_calibration_file(filename: str, dark_rate: np.ndarray, bias: np.ndarray, read_noise: np.ndarray, λ_m_qe_perc_data: tuple[np.ndarray, np.ndarray], gain: float, full_well_capacity: float, bit_depth: int):
    """Write dark_rate, bias, read_noise, and quantum_efficiency arrays to a camera calibration FITS file.

    Parameters:
        filename: str
            Destination file path.
        dark_rate: np.ndarray
            Dark current rate frame.
        bias: np.ndarray
            Bias frame.
        read_noise: np.ndarray
            Read noise frame.
        quantum_efficiency: np.ndarray
            QE table with shape (2, N): row 0 = wavelength in nm, row 1 = QE fraction.
        gain: float
            Electron count to adu conversion gain (ADU/e)
        full_well_capacity: float
            Maximum number of electrons a pixel can hold before saturation (electrons).
        bit_depth: int
            ADC bit depth, determining the number of discrete ADU levels (2**bit_depth levels).
    """
    fits_calibration = np.stack([dark_rate, bias, read_noise], axis=0)
    fits_calibration_hdu = fits.PrimaryHDU(fits_calibration)
    fits_calibration_hdu.header["FULLWELL"] = (full_well_capacity, "Full well capacity (electrons)")
    fits_calibration_hdu.header["BITDEPTH"] = (bit_depth, "ADC bit depth")
    fits_calibration_hdu.header["GAIN"] = (gain, "Conversion gain (ADU/e)")
    fits_calibration_hdu.header["NFRAME"] = 3
    fits_calibration_hdu.header["FRAME0"] = "dark_rate"
    fits_calibration_hdu.header["FRAME1"] = "bias"
    fits_calibration_hdu.header["FRAME2"] = "read_noise"
    qe_hdu = fits.BinTableHDU.from_columns([fits.Column(name="wavelength_nm", format="D", array=λ_m_qe_perc_data[0]), fits.Column(name="quantum_efficiency", format="D", array=λ_m_qe_perc_data[1])], name="QE")
    fits.HDUList([fits_calibration_hdu, qe_hdu]).writeto(filename, overwrite=True)


def is_modulator_calibration_file_valid(filename: str, shape: tuple[int, int]) -> bool:
    """Return True if the modulator calibration FITS file contains 2 frames matching the expected shape.

    Parameters:
        filename: str
            Path to the modulator calibration FITS file.
        shape: tuple[int, int]
            Expected (width, height) of each calibration frame.

    Returns: bool
        True if valid, False otherwise.
    """
    with fits.open(filename) as hdul:
        if len(hdul[0].data) != 2:
            return False
        else:
            data_shape = (shape[1], shape[0])
            return (hdul[0].data[0].shape == data_shape) & (hdul[0].data[1].shape == data_shape)


def read_modulator_calibration_file(filename: str) -> dict[str, np.ndarray]:
    """Load slope and flat arrays from a modulator calibration FITS file.

    Parameters:
        filename: str
            Path to the modulator calibration FITS file.

    Returns: dict[str, np.ndarray]
        Dict with keys slope and flat.
    """
    with fits.open(filename) as hdul:
        slope_data = hdul[0].data[0]
        flat_data = hdul[0].data[1]
        return {"slope": slope_data, "flat": flat_data}


def write_modulator_calibration_file(filename: str, slope_nm_to_adu: np.ndarray, intercept_nm: np.ndarray):
    """Write slope and flat calibration arrays to a modulator calibration FITS file.

    Parameters:
        filename: str
            Destination file path.
        slope_nm_to_adu: np.ndarray
            Pixel-wise slope converting nm deflection to ADU.
        intercept_nm: np.ndarray
            Pixel-wise flat command in ADU.
    """
    fits_calibration = np.stack([slope_nm_to_adu, intercept_nm], axis=0)
    fits_calibration_hdu = fits.PrimaryHDU(fits_calibration)
    fits_calibration_hdu.header["NFRAME"] = 2
    fits_calibration_hdu.header["FRAME0"] = "slope"
    fits_calibration_hdu.header["FRAME1"] = "flat"
    fits_calibration_hdu.writeto(filename, overwrite=True)


def read_quantum_efficiency_file(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Read quantum efficiency data from a 2-column CSV file.

    The file must have a single header row followed by rows of
    (wavelength_nm, quantum_efficiency) values.

    Parameters:
        filename: str
            Path to the CSV file.

    Returns: tuple[np.ndarray, np.ndarray]
        wavelength and quantum efficiency arrays
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def calculate_quantum_efficiency(λ: float | np.ndarray, λs: np.ndarray, qes: np.ndarray) -> float | np.ndarray:
    """Interpolate quantum efficiency at a given wavelength.

    Parameters:
        λ: float
            Query wavelength.
        λs: np.ndarray
            Wavelength sample points (must be monotonically increasing).
        qes: np.ndarray
            Quantum efficiency values corresponding to λs.

    Returns: float
        Interpolated quantum efficiency at λ.
    """
    cs = CubicSpline(λs, qes)
    return cs(λ)


def write_source_sample_header(fileio: FileIO, sample: SourceSample):
    """Write the binary file header (tag, shape, dtype) for a source sample recording.

    Parameters:
        fileio: FileIO
            Open binary file to write to.
        sample: SourceSample
            Source sample whose capture shape and dtype define the header.
    """
    h, w = sample.capture.shape[:2]
    dtype_code = DTYPE_MAP[sample.capture.dtype.type]
    header = struct.pack(
        HEADER_FORMAT,  # 7 char tag + unsigned short width + unsigned short height + unsigned byte datatype
        SRC_TAG,  # 7 char tag - 7s
        np.uint16(h),  # unsigned short height - H
        np.uint16(w),  # unsigned int width - H
        dtype_code,  # unsigned byte datatype - B
    )
    fileio.write(header)


def write_source_sample_data(fileio: FileIO, sample: SourceSample):
    """Append source sample metadata and raw capture bytes to a binary file.

    Parameters:
        fileio: FileIO
            Open binary file to write to.
        sample: SourceSample
            Source sample to serialize.
    """
    h, w = sample.capture.shape[:2]
    tl = sample.roi.get("tl", (0, 0))
    br = sample.roi.get("br", (w, h))
    sub_header = struct.pack(
        SRC_HEADER_FORMAT,  # double timestamp + double frame_rate_fps + double temperature_c + double gain + double exposure_time_s + unsigned short roi.tl.x + unsigned short roi.tl.y + unsigned short roi.br.x + unsigned short roi.br.y
        np.float64(sample.last_access_time.timestamp()),  # double timestamp - d
        np.float64(sample.frame_rate_fps),  # double frame_rate_fps - d
        np.float64(sample.temperature_c),  # double temperature_c - d
        np.float64(sample.gain),  # double gain - d
        np.float64(sample.exposure_time_s),  # double exposure_time_s - d
        np.uint16(tl[0]),  # unsigned short roi.tl.x - H
        np.uint16(tl[1]),  # unsigned short roi.tl.y - H
        np.uint16(br[0]),  # unsigned short roi.br.x - H
        np.uint16(br[1]),  # unsigned short roi.br.y - H
    )
    fileio.write(sub_header + sample.capture.tobytes())


def read_source_samples(filename: str) -> list[SourceSample]:
    """Read and return all source samples from a binary recording file.

    Parameters:
        filename: str
            Path to the binary source sample file.

    Returns: list[SourceSample]
        List of source samples in file order.
    """
    header_size = struct.calcsize(HEADER_FORMAT)
    src_header_size = struct.calcsize(SRC_HEADER_FORMAT)

    with open(filename, "rb") as rbfile:
        # --- read header ---
        header_bytes = rbfile.read(header_size)
        tag, h, w, dtype_code = struct.unpack(HEADER_FORMAT, header_bytes)
        if tag != SRC_TAG:
            raise ValueError(f"Invalid file header (tag mismatch) looking for {SRC_TAG.decode('utf-8')}, found {tag}")

        dtype = INV_DTYPE_MAP[dtype_code]
        capture_size = h * w * np.dtype(dtype).itemsize

        # --- read capture records ---
        sample_list = []
        while True:
            src_header_bytes = rbfile.read(src_header_size)
            if len(src_header_bytes) < src_header_size:
                break  # EOF
            (timestamp, frame_rate_fps, temperature_c, gain, exposure_time_s, tl_x, tl_y, br_x, br_y) = struct.unpack(SRC_HEADER_FORMAT, src_header_bytes)  # double timestamp + double frame_rate_fps + double temperature_c + double gain + double exposure_time_s + unsigned short roi.tl.x + unsigned short roi.tl.y + unsigned short roi.br.x + unsigned short roi.br.y

            capture_bytes = rbfile.read(capture_size)
            if len(capture_bytes) < capture_size:
                break  # incomplete capture

            capture = np.frombuffer(capture_bytes, dtype=dtype).reshape((h, w))
            timestamp = datetime.fromtimestamp(timestamp)

            sample_list.append(SourceSample(timestamp, exposure_time_s, gain, frame_rate_fps, temperature_c, {"tl": (tl_x, tl_y), "br": (br_x, br_y)}, capture))

        return sample_list


def write_sink_sample_header(fileio: FileIO, sample: SinkSample):
    """Write the binary file header (tag, shape, dtype) for a sink sample recording.

    Parameters:
        fileio: FileIO
            Open binary file to write to.
        sample: SinkSample
            Sink sample whose command shape and dtype define the header.
    """
    h, w = sample.command.shape[:2]
    dtype_code = DTYPE_MAP[sample.command.dtype.type]
    header = struct.pack(
        HEADER_FORMAT,  # 7 char tag + unsigned short width + unsigned short height + unsigned byte datatype
        SNK_TAG,  # 7 char tag - 7s
        np.uint16(h),  # unsigned short height - H
        np.uint16(w),  # unsigned int width - H
        dtype_code,  # unsigned byte datatype - B
    )
    fileio.write(header)


def write_sink_sample_data(fileio: FileIO, sample: SinkSample):
    """Append sink sample metadata and raw command bytes to a binary file.

    Parameters:
        fileio: FileIO
            Open binary file to write to.
        sample: SinkSample
            Sink sample to serialize.
    """
    center = sample.center
    sub_header = struct.pack(
        SNK_HEADER_FORMAT,  # double timestamp + double frame_rate_fps + unsigned short radius + unsigned short center.x + unsigned short center.y
        np.float64(sample.last_access_time.timestamp()),  # double timestamp - d
        np.float64(sample.frame_rate_fps),  # double frame_rate_fps - d
        np.uint16(sample.radius),  # unsigned short radius - H
        np.uint16(center[0]),  # unsigned short center.x - H
        np.uint16(center[1]),  # unsigned short center.y - H
    )
    fileio.write(sub_header + sample.command.tobytes())


def read_sink_samples(filename: str) -> list[SinkSample]:
    """Read and return all sink samples from a binary recording file.

    Parameters:
        filename: str
            Path to the binary sink sample file.

    Returns: list[SinkSample]
        List of sink samples in file order.
    """
    header_size = struct.calcsize(HEADER_FORMAT)
    snk_header_size = struct.calcsize(SNK_HEADER_FORMAT)

    with open(filename, "rb") as rbfile:
        # --- read header ---
        header_bytes = rbfile.read(header_size)
        tag, h, w, dtype_code = struct.unpack(HEADER_FORMAT, header_bytes)
        if tag != SNK_TAG:
            raise ValueError(f"Invalid file header (tag mismatch) looking for {SNK_TAG.decode('utf-8')}, found {tag}")

        dtype = INV_DTYPE_MAP[dtype_code]
        command_size = h * w * np.dtype(dtype).itemsize

        # --- read command records ---
        sample_list = []
        while True:
            snk_header_bytes = rbfile.read(snk_header_size)
            if len(snk_header_bytes) < snk_header_size:
                break  # EOF
            (timestamp, frame_rate_fps, radius, center_x, center_y) = struct.unpack(SNK_HEADER_FORMAT, snk_header_bytes)  # double timestamp + double frame_rate_fps + unsigned short radius + unsigned short center.x + unsigned short center.y

            command_bytes = rbfile.read(command_size)
            if len(command_bytes) < command_size:
                break  # incomplete command

            command = np.frombuffer(command_bytes, dtype=dtype).reshape((h, w))
            timestamp = datetime.fromtimestamp(timestamp)

            sample_list.append(SinkSample(timestamp, frame_rate_fps, (center_x, center_y), radius, command))

    return sample_list


def power_to_capture(power: np.ndarray, exp_time_s: float, dark_rate: np.ndarray | float = 0, bias: np.ndarray | float = 0, qe: float = 1, flat_field: float | np.ndarray = 1, gain: float = 1, full_well_capacity: float | None = None, bit_depth: int | None = None, read_noise: float | np.ndarray = 0, photon_noise: bool = False) -> np.ndarray:
    """Convert a power photons/s capture in adu.

    Parameters:
        power: np.ndarray
            Intensity in photons/s
        exp_time_s: float
            Exposure time in seconds
        dark_rate: np.ndarray | float = 0
            Dark current rate in adu/s
        bias: np.ndarray | float = 0
            Bias in adu
        qe: float = 1
            Quantum efficiency (%)
        flat_field: float | np.ndarray = 1
            Electron count to adu conversion gain (ADU/e)
        gain: float = 1
            Electron count to adu conversion gain (ADU/e)
        full_well_capacity: float | None = None
            Full well capacity in e
        bit_depth: int | None = None
            Bit depth for digitization
        read_noise: float | np.ndarray = 0
            RMS read noise in adu
        photon_noise: bool = False
            Enable photon noise

    Returns: np.ndarray
        Capture (in photon_count/electron_count/adu)
    """
    # dark current
    dark_e = exp_time_s * dark_rate / gain

    # signal in photo-electrons
    signal_e = power * exp_time_s * qe * flat_field + dark_e

    # Photon noise
    if photon_noise:
        signal_e = large_poisson(np.clip(signal_e, 0, None), thresh=1e6)

    # Saturate well
    if full_well_capacity is not None:
        signal_e = np.clip(signal_e, 0, full_well_capacity)

    # photo-electrons to adu conversion
    capture = signal_e * gain + bias

    # Add read noise
    capture = capture + np.random.normal(loc=0, scale=read_noise, size=capture.shape)

    # Saturate digitizer
    if bit_depth is not None:
        capture = np.clip(np.round(capture), 0, 2**bit_depth - 1).astype(int)

    return capture


def flip_rotate_frame(frame: np.ndarray, flip: Flip, rotation: Rotation) -> np.ndarray:
    """Apply rotation then horizontal flip to a frame array.

    Parameters:
        frame: np.ndarray
            Input frame
        flip: Flip
            Flip
        rotation: Rotation
            Rotation

    Returns: np.ndarray
        Rotated flipped frame
    """
    frame = np.rot90(frame, rotation.to_int())
    if flip == Flip.NEG:
        frame = np.fliplr(frame)
    return frame


def flip_rotate_points(xs: np.ndarray, ys: np.ndarray, image_shape: tuple[int, int], flip: Flip, rotation: Rotation) -> tuple[np.ndarray, np.ndarray]:
    """Transform point coordinates to match a flip and rotation applied to an image.

    Parameters:
        xs: np.ndarray
            x coordinates
        ys: np.ndarray
            y coordinates
        image_shape: tuple[int, int]
            Image shape
        flip: Flip
            Flip
        rotation: Rotation
            Rotation

    Returns: tuple[np.ndarray, np.ndarray]
        Rotated flipped x coordinates, Rotated flipped y coordinates
    """
    height, width = image_shape

    if rotation == Rotation.UP:  # UP
        xs_new, ys_new = xs, ys
        height_new, width_new = height, width
    elif rotation == Rotation.LEFT:  # LEFT (90° CCW)
        xs_new, ys_new = width - 1 - ys, xs
        height_new, width_new = width, height
    elif rotation == Rotation.DOWN:  # DOWN (180°)
        xs_new, ys_new = height - 1 - xs, width - 1 - ys
        height_new, width_new = height, width
    elif rotation == Rotation.RIGHT:  # RIGHT (270° CCW)
        xs_new, ys_new = ys, height - 1 - xs
        height_new, width_new = width, height

    if flip == Flip.NEG:
        ys_new = width_new - 1 - ys_new

    return xs_new, ys_new


def locate_single_airy(capture: np.ndarray, guess_radius: float) -> tuple[tuple[float, float], float, float]:
    """Locate a single Airy disk in an image by fitting an Airy function.

    Parameters:
        capture: np.ndarray
            2D image containing the Airy disk.
        guess_radius: float
            Initial guess for the Airy disk radius in pixels.

    Returns: tuple[tuple[float, float], float, float]
        Fitted center (x, y) in pixels, fitted radius in pixels, and fitted peak brightness.
    """
    y_bright, x_bright = np.unravel_index(np.argmax(capture), capture.shape)
    brightness = capture[y_bright, x_bright]

    def fit_airy_fn(xx_yy, cx, cy, r, h):
        return airy_fn(xx_yy, (cx, cy), r, h)

    (cx, cy, fit_radius, fit_height), _ = least_squares_fit_2d(capture, fit_airy_fn, guess_prms=(x_bright, y_bright, guess_radius, brightness))  # pylint: disable=unbalanced-tuple-unpacking
    fit_center = (cx, cy)
    return fit_center, fit_radius, fit_height


def locate_single_airy_with_radius(capture: np.ndarray, radius: float) -> tuple[tuple[float, float], float]:
    """Locate a single Airy disk of known radius in an image by fitting an Airy function.

    Parameters:
        capture: np.ndarray
            2D image containing the Airy disk.
        radius: float
            Known Airy disk radius in pixels (held fixed during fitting).

    Returns: tuple[tuple[float, float], float]
        Fitted center (x, y) in pixels and fitted peak brightness.
    """
    y_bright, x_bright = np.unravel_index(np.argmax(capture), capture.shape)
    brightness = capture[y_bright, x_bright]

    def fit_airy_fn(xx_yy, cx, cy, h):
        return airy_fn(xx_yy, (cx, cy), radius, h)

    (cx, cy, fit_height), _ = least_squares_fit_2d(capture, fit_airy_fn, guess_prms=(x_bright, y_bright, brightness))  # pylint: disable=unbalanced-tuple-unpacking
    fit_center = (cx, cy)
    return fit_center, fit_height


def image_shift(image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    # image_fft2 = np.fft.fft2(image)
    # image_fft2_shift = ndimage.fourier_shift(image_fft2, shift=200)
    # return np.fft.ifft2(image_fft2_shift)
    return ndimage.shift(image, shift, mode="mirror")


def pairwise_probe(shape: tuple[int, int], dξ: float, dη: float, ξc: float, θ: float, direction: PairwiseProbeDirection) -> np.ndarray:
    """Create a pairwise probe pattern image.

    Example:
        image_pairwise_probe = pairwise_probe((200,200), 0.01, 0.01, 90, 0, PairwiseProbeDirection.HORIZONTAL)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        dξ: float
            Probe rectangle size (along the PairwiseProbeDirection).
        dη: float
            Probe rectangle size (perpendicular to the PairwiseProbeDirection).
        ξc: float
            Period of the sinusoid (along the PairwiseProbeDirection).
        θ: float
            Phase of the sinusoid (along the PairwiseProbeDirection) in radians.

    Returns: np.ndarray
        Image of the pairwise probe pattern.
    """

    def _pairwise_probe(shape: tuple[int, int], dξ: float, dη: float, ξc: float, θ: float) -> np.ndarray:
        xx, yy = generate_coordinates(shape, cartesian=True, offset=(-shape[0] / 2 + 0.5, -shape[1] / 2 + 0.5))
        _2pi_xx = 2 * np.pi * xx
        _2pi_yy = 2 * np.pi * yy
        _invξc_2pi_xx = (1 / ξc) * _2pi_xx
        return np.sinc(dξ * _2pi_xx) * np.sinc(dη * _2pi_yy) * np.sin(_invξc_2pi_xx + θ)

    if direction == PairwiseProbeDirection.HORIZONTAL:
        return _pairwise_probe(shape, dξ, dη, ξc, θ)
    else:
        return np.rot90(_pairwise_probe(shape, dξ, dη, ξc, θ))


def dotf_probe(shape: tuple[int, int], size: tuple[int, int], direction: DOTFProbeDirection) -> np.ndarray:
    """Create a DOTF probe pattern image.

    Example:
        image_dotf_probe = dotf_probe((200,200), (4,11), DOTFProbeDirection.TOP)

    Parameters:
        shape: tuple[int, int]
            Image shape.
        size: tuple[int, int]
            Size of the box.
        direction: DOTFProbeDirection
            direction of the probe.

    Returns: np.ndarray
        Image of the dotf probe pattern.
    """
    width, height = shape
    a, b = size
    if direction == DOTFProbeDirection.RIGHT:
        _center = (-width, -height // 2)
        _size = a, b
    elif direction == DOTFProbeDirection.BOTTOM:
        _center = (-width // 2, 0)
        _size = b, a
    elif direction == DOTFProbeDirection.LEFT:
        _center = (0, -height // 2)
        _size = a, b
    else:  # DOTFProbeDirection.TOP
        _center = (-width // 2, -height)
        _size = b, a
    return box(shape, _size, _center)


def is_pairwise_calibration_file_valid(filename: str) -> bool:
    """Return True if the pairwise calibration pickle file contains entries for both probes 1 and 2.

    Parameters:
        filename: str
            Path to the pairwise calibration pickle file.

    Returns: bool
        True if valid, False otherwise.
    """
    with open(filename, "rb") as rbfile:
        E_star, calibration, amplitude_m, ξc, dξ, dη = cloudpickle.load(rbfile)
    if 1 not in calibration:
        return False
    if 2 not in calibration:
        return False
    return True


def read_pairwise_calibration_file(filename: str) -> tuple[float, dict[int, NDArray[np.complex128]], float, float, float, float]:
    """Load and return pairwise calibration data from a pickle file.

    Parameters:
        filename: str
            Path to the pairwise calibration pickle file.

    Returns: float, tuple[dict[int, NDArray[np.complex128]], float, float, float, float]
        (star_brightness, {1:ΔE_1, 2:ΔE_2}, amplitude_m, ξc, dξ, dη)
    """
    with open(filename, "rb") as rbfile:
        return cloudpickle.load(rbfile)


def write_pairwise_calibration_file(pairwise_calibration_dict: tuple[float, dict[int, NDArray[np.complex128]], float, float, float, float], filename: str):
    """Serialize pairwise calibration data to a pickle file.

    Parameters:
        pairwise_calibration_dict: tuple[float, tuple[dict[int, NDArray], float, float, float, float]
            Calibration data (star_brightness, {1:ΔE_1, 2:ΔE_2}, amplitude_m, ξc, dξ, dη)
        filename: str
            Destination file path.
    """
    with open(filename, "wb") as wbfile:
        cloudpickle.dump(pairwise_calibration_dict, wbfile)


def pairwise_estimate(I_p1: NDArray[np.float64], I_m1: NDArray[np.float64], I_p2: NDArray[np.float64], I_m2: NDArray[np.float64], pqrs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]) -> NDArray[np.complex128]:
    """Estimate the complex electric field using the imaginary-probe pairwise formulation.

    Parameters:
        I_p1: NDArray[np.float64]
            Intensity for positive probe 1.
        I_m1: NDArray[np.float64]
            Intensity for negative probe 1.
        I_p2: NDArray[np.float64]
            Intensity for positive probe 2.
        I_m2: NDArray[np.float64]
            Intensity for negative probe 2.
        pqrs: tuple[NDArray, NDArray, NDArray, NDArray]
            Inversion matrix elements (p, q, r, s) from pairwise_estimation_matrices.

    Returns: NDArray[np.complex128]
        Estimated complex electric field.
    """
    p, q, r, s = pqrs
    δ1 = I_p1 - I_m1
    δ2 = I_p2 - I_m2
    re_field = p * δ1 + q * δ2
    im_field = r * δ1 + s * δ2
    return 0.25 * (re_field + 1j * im_field)


def pairwise_estimation_matrices(iCAψ_1: NDArray[np.complex128], iCAψ_2: NDArray[np.complex128]) -> tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float]]:
    """Compute the inversion matrices for pairwise wavefront estimation from two complex probe fields.

    Parameters:
        iCAψ_1: NDArray[np.complex128]
            Complex probe field for probe 1.
        iCAψ_2: NDArray[np.complex128]
            Complex probe field for probe 2.

    Returns: tuple[NDArray, NDArray, NDArray, NDArray]
        Inversion matrix elements (p, q, r, s).
    """
    return invert_2x2_arrays(np.real(iCAψ_1), np.imag(iCAψ_1), np.real(iCAψ_2), np.imag(iCAψ_2))


def deflection_to_command(deflection_m: np.ndarray, slope_adu_per_m: np.ndarray, flat_adu: np.ndarray) -> np.ndarray:
    """Convert deflection in m to command in adu.

    Parameters:
        deflection_m: np.ndarray
            Deflection (m)
        slope_adu_per_m: np.ndarray
            conversion from m to adu (adu/m)
        flat_adu: np.ndarray
            flat command in (adu)

    Returns: np.ndarray
        Command (adu)
    """
    return deflection_m * slope_adu_per_m + flat_adu


def linear_fit_fn(x, m: float, c: float):
    """Evaluate a linear function: m * x + c.

    Parameters:
        x:
            Input variable.
        m: float
            Slope.
        c: float
            Intercept.

    Returns: float | np.ndarray
        Evaluated linear function.
    """
    return m * x + c


def sin_fit_fn(x, amplitude: float, frequency: float, phase: float, offset: float):
    """Evaluate amplitude * sin(frequency * x + phase) + offset.

    Parameters:
        x:
            Input variable.
        amplitude: float
            Amplitude of the sinusoid.
        frequency: float
            Angular frequency.
        phase: float
            Phase offset in radians.
        offset: float
            Vertical offset.

    Returns: float | np.ndarray
        Evaluated sinusoid.
    """
    return amplitude * np.sin(frequency * x + phase) + offset


def constrained_sin_fit_fn(x, amplitude: float, phase: float, offset: float):
    """Evaluate a unit-frequency sinusoid: amplitude * sin(x + phase) + offset.

    Parameters:
        x:
            Input variable.
        amplitude: float
            Amplitude of the sinusoid.
        phase: float
            Phase offset in radians.
        offset: float
            Vertical offset.

    Returns: float | np.ndarray
        Evaluated sinusoid.
    """
    return sin_fit_fn(x, amplitude, 1, phase, offset)


def quadratic_fit_fn(x, a: float, x0: float, c: float):
    """Evaluate a vertex-form quadratic: a * (x - x0)^2 + c.

    Parameters:
        x:
            Input variable.
        a: float
            Curvature coefficient.
        x0: float
            Vertex location.
        c: float
            Vertex value.

    Returns: float | np.ndarray
        Evaluated quadratic.
    """
    return a * (x - x0) ** 2 + c


def is_speckle_calibration_file_valid(filename: str) -> bool:
    """Return True if the speckle calibration pickle file contains the required slope/intercept keys.

    Parameters:
        filename: str
            Path to the speckle calibration pickle file.

    Returns: bool
        True if valid, False otherwise.
    """
    with open(filename, "rb") as rbfile:
        d = cloudpickle.load(rbfile)

    if "speck_angle_cmd_angle" not in d:
        return False
    if "slope" not in d["speck_angle_cmd_angle"]:
        return False
    if "intercept" not in d["speck_angle_cmd_angle"]:
        return False

    if "speck_dist_cmd_freq" not in d:
        return False
    if "slope" not in d["speck_dist_cmd_freq"]:
        return False
    if "intercept" not in d["speck_dist_cmd_freq"]:
        return False

    return True


def read_speckle_calibration_file(filename: str) -> dict[str, dict[str, float]]:
    """Load and return the speckle calibration dict from a pickle file.

    Parameters:
        filename: str
            Path to the speckle calibration pickle file.

    Returns: dict[str, dict[str, float]]
        Calibration dict with speck_dist_cmd_freq and speck_angle_cmd_angle entries.
    """
    with open(filename, "rb") as rbfile:
        return cloudpickle.load(rbfile)


def write_speckle_calibration_file(speckle_calibration_dict: dict[str, dict[str, float]], filename: str):
    """Serialize the speckle calibration dict to a pickle file.

    Parameters:
        speckle_calibration_dict: dict[str, dict[str, float]]
            Calibration dict to save.
        filename: str
            Destination file path.
    """
    with open(filename, "wb") as wbfile:
        cloudpickle.dump(speckle_calibration_dict, wbfile)


def find_speckles(speckle_image: np.ndarray, num_peaks: int = 1, footprint_size: int = 10, min_distance: int = 1) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Locate speckle peaks and return their weighted centroids sorted by x and the dilated peak mask.

    Parameters:
        speckle_image: np.ndarray
            2D intensity image to search for speckle peaks.
        num_peaks: int
            Number of speckle peaks to find.
        footprint_size: int
            Radius of the disk used to dilate each peak into a region.
        min_distance: int
            Minimum pixel separation between detected peaks.

    Returns: tuple[list[tuple[float, float]], np.ndarray]
        Weighted centroids as (x, y) pairs sorted by x, and the boolean peak mask.
    """
    peak_idx = []
    threshold_rel = 1.0
    while len(peak_idx) < num_peaks:
        peak_idx = peak_local_max(speckle_image, num_peaks=num_peaks, min_distance=min_distance, threshold_rel=threshold_rel, threshold_abs=None, exclude_border=20)
        threshold_rel = threshold_rel / 2
        if threshold_rel < 0.0625:
            break
    assert len(peak_idx) == num_peaks, f"looking for {num_peaks} speckles, found {len(peak_idx)}"
    peak_mask = np.zeros_like(speckle_image, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    disk_mask = disk(footprint_size)
    peak_mask = dilation(peak_mask, disk_mask)
    label_image = label(peak_mask)
    speckles = regionprops(label_image, speckle_image)
    ind = np.lexsort(([speckle.centroid_weighted[0] for speckle in speckles], [speckle.centroid_weighted[1] for speckle in speckles]))
    return [(speckles[i].centroid_weighted[1], speckles[i].centroid_weighted[0]) for i in ind], peak_mask


def capture_to_intensity(capture: np.ndarray, exp_time_s: float, dark_rate: np.ndarray | float = 0, bias: np.ndarray | float = 0, qe: float = 1, flat_field: float | np.ndarray = 1, gain: float = 1) -> np.ndarray:
    """Convert a capture in adu to intensity photons/s (noise-free, ignoring clipping and quantization).

    Parameters:
        capture: np.ndarray
            Raw capture in adu
        exp_time_s: float
            Exposure time in seconds
        dark_rate: np.ndarray | float = 0
            Dark current rate in adu/s
        bias: np.ndarray | float = 0
            Bias in adu
        qe: float = 1
            Quantum efficiency (%)
        flat_field: float | np.ndarray = 1
            Electron count to adu conversion gain (ADU/e)
        gain: float = 1
            Electron count to adu conversion gain (ADU/e)

    Returns: np.ndarray
        Intensity in photons/s
    """
    return (capture - bias - dark_rate * exp_time_s) / (exp_time_s * qe * gain * flat_field)


def command_to_deflection(command_adu: np.ndarray, slope_adu_per_m: np.ndarray, flat_adu: np.ndarray) -> np.ndarray:
    """Convert command in adu to deflection in m.

    Parameters:
        command: np.ndarray
            Command (adu)
        slope_adu_per_m: np.ndarray
            conversion from m to adu (adu/m)
        flat_adu: np.ndarray
            flat command in (adu)

    Returns: np.ndarray
        Deflection (m)
    """
    return (command_adu - flat_adu) / slope_adu_per_m


def create_camera_memory(name: str, full_shape: tuple[int, int], roi_shape: tuple[int, int], dtype: DataType, serial: str, pxmax: int, port: int) -> SharedMemory:
    # ---- constants ----
    kw_kind = Keyword("KIND", KeywordType.STRING, "CAMERA", "Device kind")
    kw_sn = Keyword("SN", KeywordType.STRING, serial, "Serial number")
    kw_pxmax = Keyword("PXMAX", KeywordType.LONG, int(pxmax), "Pixel max")
    kw_full_w = Keyword("FULL.W", KeywordType.LONG, int(full_shape[0]), "Detector width")
    kw_full_h = Keyword("FULL.H", KeywordType.LONG, int(full_shape[1]), "Detector height")
    kw_port = Keyword("PORT", KeywordType.LONG, int(port), "Link port")
    kw_width = Keyword("WIDTH", KeywordType.LONG, int(roi_shape[0]), "Width (px)")
    kw_height = Keyword("HEIGHT", KeywordType.LONG, int(roi_shape[1]), "Height (px)")
    # ---- variables ----
    kw_exptime = Keyword("EXPTIME", KeywordType.DOUBLE, int(0), "Exposure time (s)")
    kw_frmrate = Keyword("FRMRATE", KeywordType.DOUBLE, float(0), "Frame rate (fps)")
    kw_gain = Keyword("GAIN", KeywordType.DOUBLE, float(0), "Gain (units)")
    kw_temp = Keyword("TEMP", KeywordType.DOUBLE, float(0), "Temperature (C)")
    kw_roi_tl_x = Keyword("ROI.TL.X", KeywordType.LONG, int(0), "Region of interest top left x")
    kw_roi_tl_y = Keyword("ROI.TL.Y", KeywordType.LONG, int(0), "Region of interest top left y")
    kw_roi_br_x = Keyword("ROI.BR.X", KeywordType.LONG, int(roi_shape[0]), "Region of interest bottom right x")
    kw_roi_br_y = Keyword("ROI.BR.Y", KeywordType.LONG, int(roi_shape[1]), "Region of interest bottom right y")

    return SharedMemory.create(name, roi_shape[0] * roi_shape[1], dtype, [kw_kind, kw_sn, kw_pxmax, kw_full_w, kw_full_h, kw_port, kw_width, kw_height, kw_exptime, kw_frmrate, kw_gain, kw_temp, kw_roi_tl_x, kw_roi_tl_y, kw_roi_br_x, kw_roi_br_y])


def create_modulator_memory(name: str, full_shape: tuple[int, int], center: tuple[float, float], radius: float, dtype: DataType, serial: str, pxmax: int, port: int) -> SharedMemory:
    # ---- constants ----
    kw_kind = Keyword("KIND", KeywordType.STRING, "SLM", "Device kind")
    kw_sn = Keyword("SN", KeywordType.STRING, serial, "Serial number")
    kw_pxmax = Keyword("PXMAX", KeywordType.LONG, int(pxmax), "Pixel max")
    kw_full_w = Keyword("FULL.W", KeywordType.LONG, int(full_shape[0]), "Detector width")
    kw_full_h = Keyword("FULL.H", KeywordType.LONG, int(full_shape[1]), "Detector height")
    kw_port = Keyword("PORT", KeywordType.LONG, int(port), "Link port")
    kw_radmax = Keyword("RADMAX", KeywordType.LONG, int(radius), "Maximum Radius (px)")
    # ---- variables ----
    kw_radius = Keyword("RADIUS", KeywordType.DOUBLE, float(radius), "Radius (px)")
    kw_center_x = Keyword("CENTER.X", KeywordType.DOUBLE, float(center[0]), "Center x (px)")
    kw_center_y = Keyword("CENTER.Y", KeywordType.DOUBLE, float(center[1]), "Center y (px)")
    kw_frmrate = Keyword("FRMRATE", KeywordType.DOUBLE, float(0), "Frame rate (fps)")

    return SharedMemory.create(name, 2 * radius * 2 * radius, dtype, [kw_kind, kw_sn, kw_pxmax, kw_full_w, kw_full_h, kw_port, kw_radmax, kw_radius, kw_center_x, kw_center_y, kw_frmrate])

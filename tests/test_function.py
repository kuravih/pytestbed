import unittest

import numpy as np
from pykato.function import airy
from pykato.log import setup_logger
from pykato.plotfunction.preset import Imshow_Preset

from pytestbed.function import find_speckles, is_speckle_calibration_file_valid, read_sink_samples, read_source_samples

logger = setup_logger("test_read_functions", terminator="\n")


class TestFunction(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_read_source_samples(self):
        samples = read_source_samples("data/output/20251124.133154_capture_source.raw")
        logger.info("samples.exposure_time_s  : %d", samples.exposure_time_s)
        logger.info("samples.gain             : %f", samples.gain)
        logger.info("samples.frame_rate_fps   : %f", samples.frame_rate_fps)
        logger.info("samples.temperature_c    : %f", samples.temperature_c)
        logger.info("samples.roi              : %s", samples.roi)
        imshow_preset = Imshow_Preset(samples.captures[0])
        imshow_preset.savefig("data/plot/20251110.123116_source.png")

    def test_read_sink_samples(self):
        samples = read_sink_samples("data/output/20251124.140308_command_sink_checker.raw")
        logger.info("samples.frame_rate_fps   : %f", samples.frame_rate_fps)
        logger.info("samples.center           : %s", samples.center)
        logger.info("samples.radius           : %f", samples.radius)
        imshow_preset = Imshow_Preset(samples.commands[0])
        imshow_preset.savefig("data/plot/20251124.140308_command_sink_checker.png")

    def test_find_speckles(self):
        width, height = 512, 256
        radius = 5
        count = np.random.randint(0, 10)
        xys = np.random.uniform((radius, radius), (width - radius, height - radius), (count, 2))
        idx = np.lexsort((xys[:, 1], xys[:, 0]))
        speckle_locations_in = xys[idx]

        speckle_image = np.zeros((height, width))
        for x, y in speckle_locations_in:
            speckle_image += airy((width, height), center=(x, y), radius=0.7, height=2**16 - 1)

        speckle_locations_out, _ = find_speckles(speckle_image, num_peaks=count, footprint_size=radius)

        diagram = Imshow_Preset(speckle_image)
        for x, y in speckle_locations_out:
            diagram.get_imshow_ax().plot(x, y, "o", markersize=4, markerfacecolor="None", color="red")
        diagram.get_imshow_ax().set_xlabel("x px")
        diagram.get_imshow_ax().set_ylabel("y px")
        diagram.get_imshow_ax().set_title("Speckle Detection")

        self.assertEqual(count, len(speckle_locations_out), "Not all speckles detected")

        for (x1, y1), (x2, y2) in zip(speckle_locations_in, speckle_locations_out):
            self.assertAlmostEqual(x1, x2, places=1)
            self.assertAlmostEqual(y1, y2, places=1)

        diagram.savefig("data/plot/find_speckles.png")

    def test_is_speckle_calibration_file_valid(self):
        assert is_speckle_calibration_file_valid("data/output/speckle_calibration_instrument/20251201.155742_speckle_cal_speckle_calibration.pkl")

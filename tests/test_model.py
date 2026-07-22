import time
import unittest

import numpy as np
from pykato.log import setup_logger
from pytestbed.device import Stream
from pytestbed.device.camera import Camera
from pytestbed.device.modulator import Modulator

logger = setup_logger("test_model", terminator="\n")


class TestModel(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_simsource(self):

        camera = Camera(Stream("sim001_simsource"))
        # Properties common to all streams
        logger.info("camera.name                      : %s", camera.name)
        logger.info("camera.kind                      : %s", camera.kind)
        logger.info("camera.sn                        : %s", camera.sn)
        logger.info("camera.pxmax                     : %s", camera.pxmax)
        logger.info("camera.full_shape                : %s", camera.full_shape)
        logger.info("camera.port                      : %s", camera.port)
        logger.info("camera.metadata.creation_time    : %s", camera.creation_time)
        logger.info("camera.metadata.last_access_time : %s", camera.last_access_time)
        # Properties specific to camera streams
        logger.info("camera.exposure_time_s           : %s", camera.exposure_time_s)
        logger.info("camera.gain                      : %s", camera.gain)
        logger.info("camera.temperature_c             : %s", camera.temperature_c)
        logger.info("camera.roi                       : %s", camera.roi)
        logger.info("camera.shape                     : %s", camera.shape)
        logger.info("camera.frame_rate_fps            : %s", camera.frame_rate_fps)

    def test_simsource_capture(self):
        # source_runner.py or model_runner.py must be running
        camera = Camera(Stream("sim001_simsource"))
        for i in range(100):
            logger.info("frame : %s", i)
            print(camera.pull_capture())
            time.sleep(0.01)

    def test_simsink(self):
        slm = Modulator(Stream("sim001_simsink"))
        # Properties common to all streams
        logger.info("slm.name                         : %s", slm.name)
        logger.info("slm.kind                         : %s", slm.kind)
        logger.info("slm.sn                           : %s", slm.sn)
        logger.info("slm.pxmax                        : %s", slm.pxmax)
        logger.info("slm.full_shape                   : %s", slm.full_shape)
        logger.info("slm.port                         : %s", slm.port)
        logger.info("slm.metadata.creation_time       : %s", slm.creation_time)
        logger.info("slm.metadata.last_access_time    : %s", slm.last_access_time)
        # Properties specific to slm streams
        logger.info("slm.max_radius                   : %s", slm.max_radius)
        logger.info("slm.radius                       : %s", slm.radius)
        logger.info("slm.center                       : %s", slm.center)
        logger.info("slm.shape                        : %s", slm.shape)
        logger.info("slm.frame_rate_fps               : %s", slm.frame_rate_fps)

    def test_simsink_command(self):
        # sink_runner.py or model_runner.py must be running
        slm = Modulator(Stream("sim001_simsink"))
        for i in range(100):
            logger.info("frame : %s", i)
            command = slm.blank
            slm.push_command(command)
            time.sleep(0.01)

    def test_simsink_simsource_loop(self):
        slm = Modulator(Stream("sim001_simsink"))
        camera = Camera(Stream("sim001_simsource"))

        command = slm.blank

        slm.push_command(command.astype(np.uint16))
        time.sleep(5)
        capture = camera.pull_capture()
        print(capture)

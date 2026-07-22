import time
import unittest

import numpy as np
from pykato.function import preroll
from pykato.log import setup_logger
from pyshmio import DataType
from pytestbed.device import Stream, create_camera_memory, create_modulator_memory
from pytestbed.device.camera import Camera
from pytestbed.device.modulator import Modulator

logger = setup_logger("test_device", terminator="\n")


class TestDevice(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    # ---- source -----------------------------------------------------------------------------------------------------
    def test_source_Stream(self):
        shm_name = "stb001_stbsource"
        stream = Stream(shm_name)
        # Properties common to all streams
        logger.info("stream.name                      : %s", stream.name)
        logger.info("stream.kind                      : %s", stream.kind)
        logger.info("stream.sn                        : %s", stream.sn)
        logger.info("stream.pxmax                     : %s", stream.pxmax)
        logger.info("stream.full_shape                : %s", stream.full_shape)
        logger.info("stream.port                      : %s", stream.port)
        logger.info("stream.metadata.creation_time    : %s", stream.creation_time)
        logger.info("stream.metadata.last_access_time : %s", stream.last_access_time)
        for ikey, key in enumerate(stream.keywords):
            logger.info("keyword %02d = keywords[%8s]  : {{.value = %s,.type = %s,.comment = %s}}", ikey, key, stream.keywords[key].value, stream.keywords[key].type, stream.keywords[key].comment)

    # ---- stbsource --------------------------------------------------------------------------------------------------
    def test_stbsource(self):
        camera = Camera(Stream("stb001_stbsource"))
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

    def test_stbsource_pull_capture(self):
        # ./stbsource must be running
        camera = Camera(Stream("stb001_stbsource"))
        for i in range(100):
            logger.info("frame : %s", i)
            print(camera.pull_capture())
            time.sleep(0.01)

    # def test_stbsource_set_capture(self):
    #     # ./stbsource must be running
    #     camera = Camera(Stream("stb001_stbsource"))
    #     capture = np.zeros_like(camera.blank)
    #     camera.set_capture(capture)

    # ---- sink -------------------------------------------------------------------------------------------------------
    def test_sink_Stream(self):
        shm_name = "stb001_stbsink"
        stream = Stream(shm_name)
        # Properties common to all streams
        logger.info("stream.name                      : %s", stream.name)
        logger.info("stream.kind                      : %s", stream.kind)
        logger.info("stream.sn                        : %s", stream.sn)
        logger.info("stream.pxmax                     : %s", stream.pxmax)
        logger.info("stream.full_shape                : %s", stream.full_shape)
        logger.info("stream.port                      : %s", stream.port)
        logger.info("stream.metadata.creation_time    : %s", stream.creation_time)
        logger.info("stream.metadata.last_access_time : %s", stream.last_access_time)
        for ikey, key in enumerate(stream.keywords):
            logger.info("keyword %02d = keywords[%8s]  : {{.value = %s,.type = %s,.comment = %s}}", ikey, key, stream.keywords[key].value, stream.keywords[key].type, stream.keywords[key].comment)

    # ---- stbsink ----------------------------------------------------------------------------------------------------
    def test_stbsink(self):
        slm = Modulator(Stream("stb001_stbsink"))
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

    def test_stbsink_push_command(self):
        # ./stbsink must be running
        slm = Modulator(Stream("stb001_stbsink"))
        for i in range(100):
            logger.info("frame : %s", i)
            command = slm.blank
            slm.push_command(command)
            time.sleep(0.01)

    # def test_stbsink_get_command(self):
    #     # ./stbsink must be running
    #     slm = Modulator(Stream("stb001_stbsink"))
    #     command = slm.get_command()
    #     print(command)

    # ---- sink -------------------------------------------------------------------------------------------------------
    def test_lcdsink(self):
        slm = Modulator(Stream("lcd001_lcdsink"))
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

    def test_lcdsink_push_command(self):
        # ./lcdsink must be running
        slm = Modulator(Stream("lcd001_lcdsink"))
        for i in range(100):
            logger.info("frame : %s", i)
            slm.push_command(((2**16 - 1) * preroll(slm.shape, i, i / 100.0, 100)).astype(np.uint16))
            time.sleep(0.01)

    def test_create_camera_memory_pupil(self):
        pupil_camera_memory = create_camera_memory("pupil_camera", (512, 512), (512, 512), DataType.UINT16, "sim001", 2**16 - 1, 5001)
        logger.info("pupil_camera.creation_time             : %s", pupil_camera_memory.creation_time)
        logger.info("pupil_camera.last_access_time          : %s", pupil_camera_memory.last_access_time)
        logger.info("pupil_camera.size                      : %s", pupil_camera_memory.size)
        logger.info("pupil_camera.name                      : %s", pupil_camera_memory.name)
        for ikey, key in enumerate(pupil_camera_memory.keywords):
            logger.info("keyword %d = keywords[%s] = {.value = %s,.type = %s,.comment = %s}", ikey, key, pupil_camera_memory.keywords[key].value, pupil_camera_memory.keywords[key].type, pupil_camera_memory.keywords[key].comment)
        logger.info("pupil_camera.shape                     : %s", pupil_camera_memory.ndarray.shape)
        logger.info("pupil_camera.dtype                     : %s", pupil_camera_memory.ndarray.dtype)

    def test_create_camera_memory_image(self):
        image_camera_memory = create_camera_memory("image_camera", (256, 256), (256, 256), DataType.UINT16, "sim002", 2**16 - 1, 5002)
        logger.info("image_camera.creation_time             : %s", image_camera_memory.creation_time)
        logger.info("image_camera.last_access_time          : %s", image_camera_memory.last_access_time)
        logger.info("image_camera.size                      : %s", image_camera_memory.size)
        logger.info("image_camera.name                      : %s", image_camera_memory.name)
        for ikey, key in enumerate(image_camera_memory.keywords):
            logger.info("keyword %d = keywords[%s] = {.value = %s,.type = %s,.comment = %s}", ikey, key, image_camera_memory.keywords[key].value, image_camera_memory.keywords[key].type, image_camera_memory.keywords[key].comment)
        logger.info("image_camera.shape                     : %s", image_camera_memory.ndarray.shape)
        logger.info("image_camera.dtype                     : %s", image_camera_memory.ndarray.dtype)

    def test_create_modulator_memory_pupil(self):
        modulator_memory = create_modulator_memory("modulator", (360, 360), (180, 180), 180, DataType.UINT16, "sim003", 2**16 - 1, 5002)
        logger.info("modulator.creation_time             : %s", modulator_memory.creation_time)
        logger.info("modulator.last_access_time          : %s", modulator_memory.last_access_time)
        logger.info("modulator.size                      : %s", modulator_memory.size)
        logger.info("modulator.name                      : %s", modulator_memory.name)
        for ikey, key in enumerate(modulator_memory.keywords):
            logger.info("keyword %d = keywords[%s] = {.value = %s,.type = %s,.comment = %s}", ikey, key, modulator_memory.keywords[key].value, modulator_memory.keywords[key].type, modulator_memory.keywords[key].comment)
        logger.info("modulator.shape                     : %s", modulator_memory.ndarray.shape)
        logger.info("modulator.dtype                     : %s", modulator_memory.ndarray.dtype)

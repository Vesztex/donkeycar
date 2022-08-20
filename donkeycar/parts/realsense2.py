'''
Author: Tawn Kramer
File: realsense2.py
Date: April 14 2019
Notes: Parts to input data from Intel Realsense 2 cameras
'''
import time
import logging
import numpy as np
from donkeycar.parts import PartType, Part

logger = logging.getLogger(__name__)


class RS_T265(Part):
    '''
    The Intel Realsense T265 camera is a device which uses an imu, twin fisheye
    cameras, and an Movidius chip to do sensor fusion and emit a world space
    coordinate frame that is remarkably consistent.
    '''
    part_type = PartType.SENSE

    def __init__(self, image_output=False, calib_filename=None):
        super().__init__(image_output=image_output,
                         calib_filename=calib_filename)
        try:
            import pyrealsense2 as rs
            # Using the image_output will grab two image streams from the
            # fisheye cameras but return only one. This can be a bit much for
            # USB2, but you can try it. Docs recommend USB3 connection for
            # this.
            self.image_output = image_output

            # When we have and encoder, this will be the last vel measured.
            self.enc_vel_ms = 0.0
            self.wheel_odometer = None

            # Declare RealSense pipeline, encapsulating actual device and sensors
            print("starting T265")
            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.pose)
            profile = cfg.resolve(self.pipe)
            dev = profile.get_device()
            tm2 = dev.as_tm2()

            if self.image_output:
                # right now it's required for both streams to be enabled
                cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
                cfg.enable_stream(rs.stream.fisheye, 2) # Right camera

            if calib_filename is not None:
                pose_sensor = tm2.first_pose_sensor()
                self.wheel_odometer = pose_sensor.as_wheel_odometer()

                # calibration to list of uint8
                with open(calib_filename) as f:
                    chars = []
                    for line in f:
                        for c in line:
                            chars.append(ord(c))  # char to uint8

                # load/configure wheel odometer
                logger.info(f"loading wheel config {calib_filename}"),
                self.wheel_odometer.load_wheel_odometery_config(chars)

            # Start streaming with requested config
            self.pipe.start(cfg)
            self.running = True
            logger.warning("Warning: T265 needs a warmup period of a few seconds "
                           "before it will emit tracking data.")

            zero_vec = (0.0, 0.0, 0.0)
            self.pos = zero_vec
            self.vel = zero_vec
            self.acc = zero_vec
            self.img = None

        except ImportError:
            logger.exception(
                'RS_T265 part could not be instantiated because the '
                'pyrealsense2 python package is not installed. You can only use'
                ' this instance for checking the car app but not for running '
                'it.')

    def poll(self):
        import pyrealsense2 as rs

        if self.wheel_odometer:
            # indexed from 0, match to order in calibration file
            wo_sensor_id = 0
            frame_num = 0  # not used
            v = rs.vector()
            v.x = - self.enc_vel_ms  # m/s
            # v.z = -1.0 * self.enc_vel_ms  # m/s
            self.wheel_odometer.send_wheel_odometry(wo_sensor_id, frame_num, v)

        try:
            frames = self.pipe.wait_for_frames()
        except Exception as e:
            logger.error(e)
            return

        if self.image_output:
            # We will just get one image for now.
            # Left fisheye camera frame
            left = frames.get_fisheye_frame(1)
            self.img = np.asanyarray(left.get_data())

        # Fetch pose frame
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            self.pos = data.translation
            self.vel = data.velocity
            self.acc = data.acceleration
            logger.debug(f'realsense pos({self.pos.x}, {self.pos.y}, '
                         f'{self.pos.z})')

    def update(self):
        while self.running:
            self.poll()

    def run_threaded(self, enc_vel_ms=0):
        self.enc_vel_ms = enc_vel_ms
        return self.pos, self.vel, self.acc, self.img

    def run(self, enc_vel_ms=0):
        self.enc_vel_ms = enc_vel_ms
        self.poll()
        return self.run_threaded(enc_vel_ms)

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        self.pipe.stop()


class PosStream(Part):
    """ Helper part to split the position result of the T265 into x and y
        coordinate."""
    part_type = PartType.PERCEIVE

    def __int__(self):
        """ Creation of the PosStream part. """
        super().__init__()

    def run(self, pos):
        """
        Donkey Car interface. Split the input vector that is received from
        the pos return value of the RS_T265 part into the x and y coordinate.
        :param vector pos:  input position, is required to have x and z
                            attributes.
        :return tuple:      returns x, z attributes as x, y coordinates
        """
        # y is up, x is right, z is backwards/forwards
        return pos.x, pos.z


if __name__ == "__main__":
    c = RS_T265()
    try:
        while True:
            pos, vel, acc, img = c.run(0)
            print(pos)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    c.shutdown()

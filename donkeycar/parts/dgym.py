import os
import time
import logging
import gym
import gym_donkeycar
from donkeycar.parts import Part


logger = logging.getLogger(__name__)


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


class DonkeyGymEnv(Part):

    def __init__(self, sim_path, host="127.0.0.1", port=9091,
                 env_name="donkey-generated-track-v0",
                 conf={}, delay=0):
        """
        Creation of DonkeyGymEnv part.

        :param sim_path:
        :param host:
        :param port:
        :param env_name:
        :param conf:
        :param delay:
        """
        super().__init__(sim_path=sim_path, host=host, port=port,
                         env_name=env_name, conf=conf)

        try:
            if sim_path != "remote":
                if not os.path.exists(sim_path):
                    raise FileNotFoundError(
                        "The path you provided for the sim does not exist.")

                if not is_exe(sim_path):
                    raise FileNotFoundError("The path you provided is not an "
                                            "executable.")

            conf["exe_path"] = sim_path
            conf["host"] = host
            conf["port"] = port
            conf["guid"] = 0
            conf["frame_skip"] = 1
            self.env = gym.make(env_name, conf=conf)
            self.frame = self.env.reset()
            self.action = [0.0, 0.0, 0.0]
            self.running = True
            self.info = {'pos': (0., 0., 0.),
                         'speed': 0,
                         'cte': 0,
                         'gyro': (0., 0., 0.),
                         'accel': (0., 0., 0.),
                         'vel': (0., 0., 0.)}
            self.delay = float(delay) / 1000
            self.buffer = []
        except (ImportError, FileNotFoundError):
            logger.error("Failed to properly intitalise DonkeyGymEnv part. "
                         "You cannot run this part, but you can check it in "
                         "the car builder.")

    def delay_buffer(self, frame, info):
        now = time.time()
        buffer_tuple = (now, frame, info)
        self.buffer.append(buffer_tuple)

        # go through the buffer
        num_to_remove = 0
        for buf in self.buffer:
            if now - buf[0] >= self.delay:
                num_to_remove += 1
                self.frame = buf[1]
            else:
                break

        # clear the buffer
        del self.buffer[:num_to_remove]

    def update(self):
        while self.running:
            if self.delay > 0.0:
                current_frame, _, _, current_info = self.env.step(self.action)
                self.delay_buffer(current_frame, current_info)
            else:
                self.frame, _, _, self.info = self.env.step(self.action)

    def run_threaded(self, steering, throttle, brake=None):
        if steering is None or throttle is None:
            steering = 0.0
            throttle = 0.0
        if brake is None:
            brake = 0.0
        print(steering, throttle)

        self.action = [steering, throttle, brake]

        # Output Sim-car camera and position / imu / lidar information in a
        # tuple. Missing values from the sim will be returned as None.
        outputs = (self.frame,
                   self.info.get('pos'),
                   self.info.get('speed'),
                   self.info.get('cte'),
                   self.info.get('gyro'),
                   self.info.get('accel'),
                   self.info.get('vel'),
                   self.info.get('lidar'))
        return outputs

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.env.close()

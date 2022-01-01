import os
import time
import gym
import logging
import gym_donkeycar
from prettytable import PrettyTable
import numpy as np
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


class DonkeyGymEnv(object):

    def __init__(self, sim_path, host="127.0.0.1", port=9091,
                 headless=0, env_name="donkey-generated-track-v0",
                 sync="asynchronous", conf={}, record_location=False,
                 record_gyroaccel=False, record_velocity=False,
                 record_lidar=False, record_laps=True, delay=0, new_sim=True):

        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise Exception(
                    "The path you provided for the sim does not exist.")

            if not is_exe(sim_path):
                raise Exception("The path you provided is not an executable.")

        if new_sim:
            conf["exe_path"] = sim_path
            conf["host"] = host
            conf["port"] = port
            conf['guid'] = 0
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
        self.delay = float(delay)
        self.record_location = record_location
        self.record_gyroaccel = record_gyroaccel
        self.record_velocity = record_velocity
        self.record_lidar = record_lidar
        self.record_laps = record_laps

    def update(self):
        while self.running:
            self.frame, _, _, self.info = self.env.step(self.action)

    def run_threaded(self, steering, throttle, brake=None):
        if steering is None or throttle is None:
            steering = 0.0
            throttle = 0.0
        if brake is None:
            brake = 0.0
        if self.delay > 0.0:
            time.sleep(self.delay / 1000.0)
        self.action = [steering, throttle, brake]

        # Output Sim-car position and other information if configured
        outputs = [self.frame]
        if self.record_location:
            outputs.append(self.info['pos'])
            outputs += self.info['speed'], self.info['cte']
        if self.record_gyroaccel:
            outputs.append(self.info['gyro'])
            outputs.append(self.info['accel'])
        if self.record_velocity:
            outputs.append(self.info['vel'])
        if self.record_lidar:
            outputs.append(self.info['lidar'])
        if self.record_laps:
            outputs.append(self.info['last_lap_time'])
        if len(outputs) == 1:
            return self.frame
        else:
            return outputs

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.env.close()


class GymLapTimer:
    def __init__(self):
        self.current_lap = 0
        self.last_lap_time = 0.0
        self.lap_times = []
        self.last_lap_distance = 0.0
        self.lap_lengths = []
        logger.info("Create GymLapTimer")

    def run(self, last_lap_time, distance):
        if last_lap_time != self.last_lap_time:
            logger.info(f'Recorded lap {self.current_lap} in '
                        f'{last_lap_time: 6.2f}s')
            # update current lap
            self.current_lap += 1
            self.lap_times.append(last_lap_time)
            lap_length = distance - self.last_lap_distance
            self.lap_lengths.append(lap_length)
            # store current values
            self.last_lap_time = last_lap_time
            self.last_lap_distance = distance
        return self.current_lap

    def shutdown(self):
        logger.info('All lap times:')
        pt = PrettyTable()
        pt.field_names = ["lap", "time (s)"]
        for i, lap in enumerate(self.lap_times):
            row = [str(i), f'{lap: 4.2f}']
            pt.add_row(row)
        logger.info('\n' + str(pt))

    def to_list(self):
        info = [dict(lap=i, time=t, distance=l) for i, (t, l) in
                enumerate(zip(self.lap_times, self.lap_lengths))]
        return info


class GymOdometer:
    def __init__(self):
        self.last_coordinate = None
        self.total_distance = 0.0
        logger.info("Create GymLapOdometer")

    def run(self, coordinate):
        if self.last_coordinate is None:
            self.last_coordinate = np.array(coordinate)
        else:
            this_coordinate = np.array(coordinate)
            dist = norm(this_coordinate - self.last_coordinate)
            self.last_coordinate = this_coordinate
            self.total_distance += dist
        return self.total_distance

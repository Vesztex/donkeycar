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
                 record_lidar=False, record_laps=True, delay=0, new_sim=True,
                 respawn_on_game_over=False):

        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise Exception(
                    "The path you provided for the sim does not exist.")

            if not is_exe(sim_path):
                raise Exception("The path you provided is not an executable.")

        if new_sim or sim_path == 'remote':
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
        self.delay = float(delay) / 1000
        self.record_location = record_location
        self.record_gyroaccel = record_gyroaccel
        self.record_velocity = record_velocity
        self.record_lidar = record_lidar
        self.record_laps = record_laps
        self.respawn_on_game_over = respawn_on_game_over
        self.done = False

        self.buffer = []

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
            self.frame, reward, self.done, self.info = \
                self.env.step(self.action)
            if self.delay > 0.0:
                self.delay_buffer(self.frame, self.info)

    def run_threaded(self, steering, throttle, brake=None):
        if steering is None or throttle is None:
            steering = 0.0
            throttle = 0.0
        if brake is None:
            brake = 0.0

        self.action = [steering, throttle, brake]
        game_over = self.done
        if self.done:
            if self.respawn_on_game_over:
                logger.info('Game Over')
                self.env.reset()
                self.done = False

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
            outputs.append(self.info.get('last_lap_time', 0.0))
            outputs.append(game_over)
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
        self.last_lap_distance = 0.0
        self.lap_info = []
        logger.info("Create GymLapTimer")

    def run(self, last_lap_time, distance):
        if last_lap_time != self.last_lap_time:
            # invalid loops in the sim have lap time = 0.0
            is_valid = last_lap_time > 0.1
            lap_length = distance - self.last_lap_distance
            logger.info(f'Recorded {"in" if not is_valid else ""}valid lap'
                        f' {self.current_lap} in '
                        f'{last_lap_time:.2f}s of length {lap_length:.2f}m')
            # store current values
            self.last_lap_time = last_lap_time
            self.last_lap_distance = distance

            info = dict(lap=self.current_lap, time=last_lap_time,
                        distance=lap_length, valid=is_valid)
            self.lap_info.append(info)
            # update current lap and reset valid lap
            self.current_lap += 1

        return self.current_lap

    def shutdown(self):
        logger.info('All lap times:')
        pt = PrettyTable()
        pt.field_names = ["lap", "time (s)", "distance", "valid"]
        valid_laps = []
        for info in self.lap_info:
            row = [info["lap"], f'{info["time"]: 4.2f}',
                   f'{info["distance"]: 5.3f}', info["valid"]]
            pt.add_row(row)
            if info["valid"]:
                valid_laps.append(info["time"])
        laps_sorted = sorted(valid_laps)
        logger.info('\n' + str(pt))
        logger.info('Three fastest laps: '
                    + ",".join(f"{it:.2f}" for it in laps_sorted[:3]))
        logger.info(f'Mean lap time: {sum(laps_sorted)/len(laps_sorted):.2f}, '
                    f'median lap time: {laps_sorted[int(len(laps_sorted)/2)]}')

    def to_list(self):
        return self.lap_info


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

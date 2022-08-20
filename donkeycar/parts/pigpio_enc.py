import logging
import time
from donkeycar.parts import Part, PartType

logger = logging.getLogger(__name__)


class OdomDist(object):
    """
    Take a tick input from odometry and compute the distance travelled
    """
    def __init__(self, mm_per_tick, debug=False):
        self.mm_per_tick = mm_per_tick
        self.m_per_tick = mm_per_tick / 1000.0
        self.meters = 0
        self.last_time = time.time()
        self.meters_per_second = 0
        self.debug = debug
        self.prev_ticks = 0

    def run(self, ticks, throttle):
        """
        inputs => total ticks since start
        inputs => throttle, used to determine positive or negative vel
        return => total dist (m), current vel (m/s), delta dist (m)
        """
        new_ticks = ticks - self.prev_ticks
        self.prev_ticks = ticks

        # save off the last time interval and reset the timer
        start_time = self.last_time
        end_time = time.time()
        self.last_time = end_time
        
        # calculate elapsed time and distance traveled
        seconds = end_time - start_time
        distance = new_ticks * self.m_per_tick
        if throttle < 0.0:
            distance = distance * -1.0
        velocity = distance / seconds
        
        # update the odometer values
        self.meters += distance
        self.meters_per_second = velocity

        # console output for debugging
        if self.debug:
            print('seconds:', seconds)
            print('delta:', distance)
            print('velocity:', velocity)

            print('distance (m):', self.meters)
            print('velocity (m/s):', self.meters_per_second)

        return self.meters, self.meters_per_second, distance


class PiGPIOEncoder(Part):
    """
    Encoder part for odometry. This part will only work on RPi because it
    requires the pigpio library. The part simply counts the number of encoder
    ticks since part creation.
    """
    part_type = PartType.SENSE

    def __init__(self, pin):
        """
        Creation of PiGPIOEncoder.

        :param int pin:     Input pin to which the encoder is connected
        """
        super().__init__(pin=pin)
        import pigpio
        try:
            self.pin = pin
            self.pi = pigpio.pi()
            self.pi.set_mode(pin, pigpio.INPUT)
            self.pi.set_pull_up_down(pin, pigpio.PUD_UP)
            self.cb = pi.callback(self.pin, pigpio.FALLING_EDGE, self._cb)
            self.count = 0
        except ImportError as e:
            logging.error('PiGPIOEncoder part could not be instantiated as the '
                          'pigpio python package is not installed. You can '
                          'only use this instance for checking the car app '
                          'but not for running it.')

    def _cb(self, pin, level, tick):
        self.count += 1

    def run(self):
        """
        Donkey car interface.

        :return int:    Returns the number of encoder counts
        """
        return self.count

    def shutdown(self):
        if self.cb is not None:
            self.cb.cancel()
            self.cb = None
        self.pi.stop()


if __name__ == "__main__":
    e = PiGPIOEncoder(4)
    while True:
        time.sleep(0.1)
        e.run()



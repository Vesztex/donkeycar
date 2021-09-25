import time
from threading import Thread, active_count

import RPi.GPIO as GPIO
from donkeycar.parts.actuator import PCA9685
import logging
from collections import deque
import queue


logger = logging.getLogger(__name__)


class LED:
    ''' 
    Toggle a GPIO pin for led control
    '''
    def __init__(self, pin):
        self.pin = pin

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT)
        self.blink_changed = 0
        self.on = False

    def toggle(self, condition):
        if condition:
            GPIO.output(self.pin, GPIO.HIGH)
            self.on = True
        else:
            GPIO.output(self.pin, GPIO.LOW)
            self.on = False            

    def blink(self, rate):
        if time.time() - self.blink_changed > rate:
            self.toggle(not self.on)
            self.blink_changed = time.time()

    def run(self, blink_rate):
        if blink_rate == 0:
            self.toggle(False)
        elif blink_rate > 0:
            self.blink(blink_rate)
        else:
            self.toggle(True)

    def shutdown(self):
        self.toggle(False)        
        GPIO.cleanup()


class RGB_LED:
    ''' 
    Toggle a GPIO pin on at max_duty pwm if condition is true, off if condition is false.
    Good for LED pwm modulated
    '''
    def __init__(self, pin_r, pin_g, pin_b, invert_flag=False):
        self.pin_r = pin_r
        self.pin_g = pin_g
        self.pin_b = pin_b
        self.invert = invert_flag
        print('setting up gpio in board mode')
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin_r, GPIO.OUT)
        GPIO.setup(self.pin_g, GPIO.OUT)
        GPIO.setup(self.pin_b, GPIO.OUT)
        freq = 50
        self.pwm_r = GPIO.PWM(self.pin_r, freq)
        self.pwm_g = GPIO.PWM(self.pin_g, freq)
        self.pwm_b = GPIO.PWM(self.pin_b, freq)
        self.pwm_r.start(0)
        self.pwm_g.start(0)
        self.pwm_b.start(0)
        self.zero = 0
        if( self.invert ):
            self.zero = 100

        self.rgb = (50, self.zero, self.zero)

        self.blink_changed = 0
        self.on = False

    def toggle(self, condition):
        if condition:
            r, g, b = self.rgb
            self.set_rgb_duty(r, g, b)
            self.on = True
        else:
            self.set_rgb_duty(self.zero, self.zero, self.zero)
            self.on = False

    def blink(self, rate):
        if time.time() - self.blink_changed > rate:
            self.toggle(not self.on)
            self.blink_changed = time.time()

    def run(self, blink_rate):
        if blink_rate == 0:
            self.toggle(False)
        elif blink_rate > 0:
            self.blink(blink_rate)
        else:
            self.toggle(True)

    def set_rgb(self, r, g, b):
        r = r if not self.invert else 100-r
        g = g if not self.invert else 100-g
        b = b if not self.invert else 100-b
        self.rgb = (r, g, b)
        self.set_rgb_duty(r, g, b)

    def set_rgb_duty(self, r, g, b):
        self.pwm_r.ChangeDutyCycle(r)
        self.pwm_g.ChangeDutyCycle(g)
        self.pwm_b.ChangeDutyCycle(b)

    def shutdown(self):
        self.toggle(False)
        GPIO.cleanup()


# Colors
RED = (4095, 0, 0)
GREEN = (0, 4095, 0)
BLUE = (0, 0, 4095)
YELLOW = (4095, 1024, 0)
PURPLE = (1024, 0, 4095)
WHITE = (4095, 1048, 4095)
OFF = (0, 0, 0)


class LEDStatus:
    def __init__(self, r_channel=13, g_channel=14, b_channel=15):
        self.rgb_pins \
            = (PCA9685(r_channel), PCA9685(g_channel), PCA9685(b_channel))
        self.pwm = [None, None, None]
        self.run = True
        # frequency, usually 60
        self.f = self.rgb_pins[0].default_freq
        self.delay = 4
        self.queue = queue.Queue()
        self.continuous = None
        self.continuous_loop = True
        self.larsen(3)
        logger.info("Created LEDStatus part")

    def _set_color(self, color):
        for i, (c, pin) in enumerate(zip(color, self.rgb_pins)):
            pin.set_pulse(c)
            self.pwm[i] = c

    def pulse(self):
        """ Produces pulsed or blinking continuous signal """
        while self.continuous_loop:
            self.blink(4 * self.delay, GREEN, 1)

        # end of thread switch off light
        self._set_color(OFF)

    def blink(self, delay, color, num, short=True):
        """
        Blinks pin n x times
        :param delay:   How fast
        :param color:   rgb tuple in [0, 4095]
        :param num:     How often
        :param short:   If short on long off or vice versa
        :return:        None
        """
        on_time = delay / self.f
        off_time = 2 * on_time
        if not short:
            on_time, off_time = off_time, on_time
        for _ in range(num):
            self._set_color(color)
            time.sleep(on_time)
            self._set_color(OFF)
            time.sleep(off_time)

    def larsen(self, num):
        on_time = 0.2
        colors = (BLUE, GREEN, RED)
        for _ in range(num):
            for col in colors:
                self._set_color(col)
                time.sleep(on_time)
        self._set_color(OFF)

    def _start_continuous(self):
        logger.debug('Starting continuous...')
        self.continuous_loop = True
        self.continuous = Thread(target=self.pulse, daemon=True)
        self.continuous.start()
        logger.debug('... started')

    def _stop_continuous(self):
        logger.debug('Stopping continuous...')
        self.continuous_loop = False
        self.continuous.join()
        logger.debug('... stopped')

    def update(self):
        # start the continuous thread
        self._start_continuous()
        # this is the queue worker
        while self.run:
            i = self.queue.get()
            # stop continuous pulsing
            self._stop_continuous()
            # show incoming signals
            i.start()
            i.join()
            # restart continuous pulsing
            self._start_continuous()

    def run_threaded(self, mode=None, speed=None, lap=False, wipe=False):
        if mode is not None:
            new_pulse = mode < 1

        if lap:
            # 3 red blinks when lap
            t = Thread(target=self.blink, args=(6, RED, 3))
            self.queue.put(t)
        if wipe:
            # 1 violet blink when wiper
            t = Thread(target=self.blink, args=(30, PURPLE, 1, False))
            self.queue.put(t)

    def shutdown(self):
        # stop the loop
        self.run = False
        self._stop_continuous()
        self.blink(20, WHITE, 2, False)


if __name__ == "__main__":
    import time
    import sys
    pin_r = int(sys.argv[1])
    pin_g = int(sys.argv[2])
    pin_b = int(sys.argv[3])
    rate = float(sys.argv[4])
    print('output pin', pin_r, pin_g, pin_b, 'rate', rate)

    p = RGB_LED(pin_r, pin_g, pin_b)
    
    iter = 0
    while iter < 50:
        p.run(rate)
        time.sleep(0.1)
        iter += 1
    
    delay = 0.1

    iter = 0
    while iter < 100:
        p.set_rgb(iter, 100-iter, 0)
        time.sleep(delay)
        iter += 1
    
    iter = 0
    while iter < 100:
        p.set_rgb(100 - iter, 0, iter)
        time.sleep(delay)
        iter += 1

    p.shutdown()


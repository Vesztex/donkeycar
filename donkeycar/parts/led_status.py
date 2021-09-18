import time
from threading import Thread

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


class LEDStatus:
    def __init__(self, r_channel=13, g_channel=14, b_channel=15):
        self.r_pin = PCA9685(r_channel)
        self.g_pin = PCA9685(g_channel)
        self.b_pin = PCA9685(b_channel)
        self.run = False
        # frequency, usually 60
        self.f = self.r_pin.default_freq
        self.speed = 4
        self.is_pulse = True
        self.queue = queue.Queue()
        # 12-bit range, so 12-14 will give full illumination
        self.pulse_pwm = [min(2 ** i - 1, 4095) for i in range(15)]
        self.pulse_pwm += list(reversed(self.pulse_pwm))
        self.continuous = None
        self.continuous_run = True
        self.larsen(2)
        logger.info("Created LEDStatus part")

    def pulse(self):
        """ Produces pulsed or blinking continuous signal """
        while self.continuous_run:
            if self.is_pulse:
                for i in self.pulse_pwm:
                    self.g_pin.set_pulse(i)
                    time.sleep(self.speed / self.f)
            else:
                self.blink(4 * self.speed, self.g_pin, 1)

    def blink(self, speed, pin, num):
        """
        Blinks pin n x times
        :param speed:   How fast
        :param pin:     Which pin, r, g or b
        :param num:     How often
        :return:        None
        """
        on_time = speed / self.f
        for _ in range(num):
            pin.set_pulse(4095)
            time.sleep(on_time)
            pin.set_pulse(0)
            time.sleep(2 * on_time)

    def larsen(self, num):
        on_time = 0.25
        pins = deque((self.r_pin, self.g_pin, self.b_pin))
        for _ in range(3 * num):
            pins[0].set_pulse(4095)
            pins[1].set_pulse(0)
            pins[2].set_pulse(0)
            pins.rotate()
            time.sleep(on_time)

    def _start_continuous(self):
        self.continuous_run = True
        self.continuous = Thread(target=self.pulse, daemon=True)
        self.continuous.start()

    def _stop_continuous(self):
        self.continuous_run = False
        self.continuous.join()

    def update(self):
        # start the continuous thread to drive pulsing/blinking signal
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

    def run_threaded(self, on, speed=None, lap=None, wipe=None):
        if on:
            if not self.continuous.is_alive():
                logger.info('Starting continuous')
                self._start_continuous()
        else:
            if self.continuous.is_alive:
                logger.info('Stopping continuous')
                self._stop_continuous()

    def shutdown(self):
        # stop the loop
        self.run = False
        self.continuous_run = False
        self.larsen(2)


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


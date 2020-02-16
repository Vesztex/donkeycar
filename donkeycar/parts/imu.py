import time
import board
import busio
import adafruit_mpu6050
import numpy as np


class Mpu6050:
    '''
    Installation:
    sudo apt install python3-smbus
    or
    sudo apt-get install i2c-tools libi2c-dev python-dev python3-dev
    git clone https://github.com/pimoroni/py-smbus.git
    cd py-smbus/library
    python setup.py build
    sudo python setup.py install

    pip install mpu6050-raspberrypi
    '''

    def __init__(self, addr=0x68, poll_delay=0.0166):
        from mpu6050 import mpu6050
        self.sensor = mpu6050(addr)
        self.accel = None
        self.gyro = None
        self.temp = None
        self.poll_delay = poll_delay
        self.on = True
        # initial call, first values seem rubbish
        self.poll()
        # now wait...
        time.sleep(self.poll_delay)
        # ... and poll again
        self.poll()
        # use these values as zero
        self.accel_zero = np.array(list(self.accel.values()))
        self.gyro_zero = np.array(list(self.gyro.values()))

    def update(self):
        while self.on:
            self.poll()
            time.sleep(self.poll_delay)

    def poll(self):
        try:
            self.accel, self.gyro, self.temp = self.sensor.get_all_data()
        except OSError as e:
            print('Failed to read imu: ', e)
            
    def run_threaded(self):
        return np.array(list(self.accel.values())) - self.accel_zero, \
               np.array(list(self.gyro.values())) - self.gyro_zero

    def run(self):
        self.poll()
        return self.run_threaded()

    def shutdown(self):
        self.on = False


class Mpu6050Ada:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.mpu = adafruit_mpu6050.MPU6050(i2c)
        self.mpu.accelerometer_range = adafruit_mpu6050.Range.RANGE_2_G
        self.mpu.gyro_range = adafruit_mpu6050.GyroRange.RANGE_250_DPS
        self.accel_zero = np.array(self.mpu.acceleration)
        self.gyro_zero = np.array(self.mpu.gyro)
        self.accel = np.zeros((3,))
        self.gyro = np.zeros((3,))

    def run(self):
        return np.array(self.mpu.acceleration) - self.accel_zero, \
               np.array(self.mpu.gyro) - self.gyro_zero


if __name__ == "__main__":
    count = 0
    p = Mpu6050Ada()
    while True:
        try:
            data = p.run()
            print(data)
            time.sleep(0.05)
            count += 1
        except KeyboardInterrupt:
            break



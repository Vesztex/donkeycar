import time


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
        self.accel_zero = self.accel
        self.gyro_zero = self.gyro

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
        return self._diff(self.accel, self.accel_zero), \
               self._diff(self.gyro, self.gyro_zero), \
               self.temp

    def run(self):
        self.poll()
        return self.run_threaded()

    def shutdown(self):
        self.on = False

    @staticmethod
    def _subtract(d1, d2):
        """ requires equal keys in d1, d2 which is not checked """
        d = d1
        for k, v in d1.iteritems():
            v -= d2[k]
        return d


if __name__ == "__main__":
    count = 0
    p = Mpu6050()
    while count < 100:
        data = p.run()
        print(data)
        time.sleep(0.1)
        count += 1

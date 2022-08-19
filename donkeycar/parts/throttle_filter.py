from donkeycar.parts import Part


class ThrottleFilter(Part):
    """
    This part modifies the raw input throttle value to allow reverse to
    trigger automatic reverse throttle. Reverse throttle on a car ESC usually
    just triggers the brake and only after going into stop it will go into
    reverse at the next push of the throttle.
    """

    def __init__(self):
        """Creation of the part"""
        self.reverse_triggered = False
        self.last_throttle = 0.0

    def run(self, throttle_in):
        """
        Donkey car run interface. We read a throttle-in value and return a
        throttle-out. If throttle-in is negative, the part will convert it to a
        throttle-stop (i.e 0.0) and a throttle negative consecutively.

        :param float throttle_in:   input throttle value in [-1, 1]
        :return float:              output throttle value in [-1, 1]
        """
        if throttle_in is None:
            return throttle_in

        throttle_out = throttle_in

        if throttle_out < 0.0:
            if not self.reverse_triggered and self.last_throttle < 0.0:
                throttle_out = 0.0
                self.reverse_triggered = True
        else:
            self.reverse_triggered = False

        self.last_throttle = throttle_out
        return throttle_out

    def shutdown(self):
        pass

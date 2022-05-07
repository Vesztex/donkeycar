from donkeycar.parts.part import Creatable
from donkeycar.utilities.deprecated import deprecated


@deprecated("This part is not used anywhere")
class Pipeline:
    def __init__(self, steps):
        super().__init__(steps=steps)
        self.steps = steps

    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']

            val = f(val, *args, **kwargs)
        return val


class Dispatcher(Creatable):
    """
    This part is a generic dispatcher that reads an index i and a tuple of
    arguments and then returns the i'th argument. This is often used in the
    car app for example when we want to switch between user/throttle and
    pilot/throttle depending on another user input, for example the web
    controller button or a remote control button.
    """
    def __init__(self, num_args=2):
        super().__init__(num_args=num_args)
        self.num_args = num_args

    def run(self, index=0, *args):
        """
        Donkeycar parts interface
        :param int index:   index of argument to return
        :param tuple args:  tuple of arguments
        :return:            chosen argument
        """
        assert len(args) == self.num_args, \
            f"Expected {self.num_args} arguments, but got {len(args)}"
        assert int(index) < self.num_args, \
            f"Can only use index < {self.num_args}"
        return args[int(index)]


class Checker(Creatable):
    """
    This part is a generic variable checker that checks if an input variable
    at runtime matches the value of the value given in the constructor or at
    least one entry if a list is given to the constructor.
    """
    def __init__(self, must_match=None):
        super().__init__(must_match=must_match)
        self.must_match = must_match

    def run(self, input):
        """
        :param Any input:   An input value
        :return bool:       True if the value equals the saved value or is in
                            the iterable of the saved values
        """
        return input in self.must_match if type(self.must_match) is list \
            else input == self.must_match
from donkeycar.parts.part import Creatable, PartType
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
    arguments and then returns the i'th argument. This logic is used in the
    car app for example when we want to switch between user/throttle and
    pilot/throttle depending on another user input, for example the web
    controller button or a remote control button.
    """
    part_type = PartType.PROCESS

    def __init__(self, num_args=2):
        """
        Creating part Dispatcher

        :param int num_args: expected number of arguments, defaults to two.
        """
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


class MultiDispatcher(Creatable):
    """
    This part is a generic dispatcher that is constructed with a list of
    keys. In the vehicle loop it compares the first input to the elements of
    the key list and finds its index n. Then it returns the n'th input of the
    remaining arguments.

    For example:

    m = MultiDispatcher(keys=['cat', 'dog', 'bird'])
    output = m.run('dog', input_0, input_1, input_2)

    will return input_1 as output, because 'dog' is the second element of the
    key list, i.e. it has index 1. Note, the key needs to match one list
    element, otherwise an error is thrown.

    Alternatively the Dispatcher can be initialised without any keys and the
    index can be passed directly to the run function:

    m = Dispatcher()
    output = m.run(2, input_0, input_1, input_2)

    will return input_2 as output. Note the index needs to be smaller than
    the number of following arguments.

    This dispatcher logic is used in the car app for switching between
    user/throttle and pilot/throttle depending on another user input,
    like the web controller mode or a remote control button.
    """
    part_type = PartType.PROCESS

    def __init__(self, keys=None):
        """
        Creating part MultiDispatcher

        :param int num_args: expected number of arguments, defaults to two.
        """
        super().__init__(keys=keys)
        self.keys = keys

    def run(self, key, *args):
        """
        The method returns a single argument of the *args tuple that is
        passed in. The selection depends on the key argument passed in first.
        If the Dispatcher part has been initialised with a list, then the key
        will be found in the list and the argument of the matching index is
        returned. If key is an integer, the corresponding argument will be
        returned. Note, counting starts at zero.

        :param int/any key:     Either index of argument to return, or key to
                                compare against key list passed in construction.
        :param tuple args:      Tuple of arguments
        :return:                Chosen argument
        """
        # multiple asserts first
        if isinstance(key, (int, bool)):
            assert self.keys == None, \
                "Cannot call run() method with an integer or bool index, " \
                "because Dispatcher has been created with a key list."
            # cast to int, in case it is a bool
            index = int(key)
            assert index < len(args)

        assert len(args) == self.num_args, \
            f"Expected {self.num_args} arguments, but got {len(args)}"
        assert int(index) < self.num_args, \
            f"Can only use index < {self.num_args}"
        return args[int(index)]


class Checker(Creatable):
    """
    This part is a generic variable checker that checks if an input variable
    at runtime matches the value (or any value if it is a list) given in the
    constructor
    """
    part_type = PartType.PROCESS

    def __init__(self, must_match=None):
        """
        Create part Checker

        :param Any/list must_match: value or list of values which the input
                                    will be compared against.
        """
        super().__init__(must_match=must_match)
        self.must_match = must_match

    def run(self, input):
        """
        This method return True if the input value matches the saved value (
        or one of the saved values) given in the constructor.

        :param Any input:   An input value
        :return bool:       True if the value equals the saved value or is in
                            the iterable of the saved values
        """
        return input in self.must_match if type(self.must_match) is list \
            else input == self.must_match

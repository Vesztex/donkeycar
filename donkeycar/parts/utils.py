from donkeycar.parts.part import Part, PartType
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


class Dispatcher(Part):
    """
    This part is a generic dispatcher that is constructed with a list of
    keys. In the vehicle loop's run method, it compares the first input to
    the elements of the key list and finds its index n. Then it returns the
    n'th input of the remaining arguments.

    For example:

    m = Dispatcher(keys=['cat', 'dog', 'bird'])
    output = m.run('dog', input_0, input_1, input_2)

    will return input_1 as output, because 'dog' is the second element of the
    key list, i.e. it has index 1. As usual, array indexes start at zero.
    Note, the key needs to match one list element, otherwise an error is
    thrown.

    Alternatively the Dispatcher can be initialised without any keys and the
    index can be passed directly to the run function:

    m = Dispatcher()
    output = m.run(2, input_0, input_1, input_2)

    will return input_2 as output. Note the index needs to be smaller than
    the number of following arguments.

    This dispatcher logic is typically used in the car app for switching
    between user/throttle and pilot/throttle depending on another user input,
    like the web controller mode or a remote control button.
    """
    part_type = PartType.PROCESS

    def __init__(self, keys=None):
        """
        Creating part Dispatcher

        :param list keys:   List of expected keys to compare against.
                            Optional parameter, if not provided, then the run
                            method will require an integer as the first
                            argument.
        """
        if keys is not None:
            assert isinstance(keys, list), f"Input keys can only be None or " \
                                           f"list but it was {keys}"
            assert len(keys) > 1, \
                f'Input list needs more than one entries, but has {len(list)}'
        super().__init__(keys=keys)
        self.keys = keys

    def run(self, key, *args):
        """
        The method returns a single argument of the *args tuple that is passed
        in. The selection depends on the key argument passed in first. If the
        Dispatcher part has been initialised with a list, then the key will be
        found in the list and the argument of the matching index is returned. If
        key is an integer, the corresponding argument will be returned. Note,
        counting starts at zero.

        :param int/any key:     Either index of argument to return, or key to
                                compare against key list passed in construction.
        :param tuple args:      Tuple of arguments
        :return any:            Input argument at position of key
        """
        # if key is int or bool
        if isinstance(key, (int, bool)):
            assert self.keys is None, \
                "Cannot call run() method with an integer or bool index, " \
                "because Dispatcher has been created with a key list."
            # cast to int, in case it is a bool
            index = int(key)
            assert index < len(args)
        # otherwise key must be in list (and is usually a string)
        else:
            index = self.keys.index(key)
        return args[index]


class Checker(Part):
    """
    This part is a generic variable checker that checks if an input variable
    at runtime matches the value (or any value if it is a list) given in the
    constructor
    """
    part_type = PartType.PROCESS

    def __init__(self, must_match=None, match_true=True):
        """
        Create part Checker

        :param Any/list must_match: Value or list of values which the input
                                    will be compared against.
        :param bool match_true:     Switch that indicates if the match
                                    outcome should be inverted. Defaults
                                    to False.
        """
        assert type(match_true) is bool, "match_true parameter must be bool"
        super().__init__(must_match=must_match, match_true=match_true)
        self.must_match = must_match
        self.match_true = match_true

    def run(self, input):
        """
        This method return True if the input value matches the saved value (
        or one of the saved values) given in the constructor.

        :param Any input:   An input value
        :return bool:       Returns match_true if the value equals the saved
                            value or is in the iterable of the saved values
        """
        result = input in self.must_match if type(self.must_match) is list \
            else input == self.must_match
        return result == self.match_true



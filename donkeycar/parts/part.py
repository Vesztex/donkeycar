from donkeycar.config import Config


class CreatableFactory(type):
    """
    Metaclass to hold the registration dictionary of the part creation function
    """
    register = {}

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        l_name = name.lower()
        # don't register base class constructor
        if l_name != 'creatable':
            cls.register[l_name] = cls.create

    @classmethod
    def make(mcs, concrete, cfg, kwargs):
        return mcs.register[concrete.lower()](cfg, kwargs)


class Creatable(object, metaclass=CreatableFactory):
    """
    Base class for factory creatable parts, implementing create() by calling
    constructor without any config
    """
    @classmethod
    def create(cls, cfg, kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestCreatable(Creatable):
    """
    Test Part that shows the creation of new parts
    """
    def __init__(self, value=2):
        self.value = value
        print('Created TestCreatable with value', self.value)


if __name__ == '__main__':
    # we allow any case for the part in the dictionary, as python classes are
    # expected to be camel case
    data = [{'testcreatable': None},
            {'TestCreatable': {'arguments': {'value': 4}}}]
    cfg = Config()

    for d in data:
        for k, v in d.items():
            args = {}
            if type(v) is dict and 'arguments' in v:
                args = v['arguments']
            part = CreatableFactory.make(k, cfg, args)


class CreatableFactory(type):
    """
    Metaclass to hold the registration dictionary of the part constructor
    functions
    """
    register = {}

    def __init__(cls, name, bases, dct):
        l_name = name.lower()
        # don't register base class constructor
        if l_name != 'creatable':
            cls.register[l_name] = cls.create

    @classmethod
    def make(mcs, concrete, kwargs):
        return mcs.register[concrete.lower()](kwargs)


class Creatable(object, metaclass=CreatableFactory):
    """
    Base class for factory creatable parts, implementing create()
    """
    @classmethod
    def create(cls, kwargs):
        return cls(**kwargs)


class TestCreatable(Creatable):
    """
    Test Part that shows the creation of new parts
    """
    def __init__(self, value):
        self.value = value
        print('Created TestCreatable with value', self.value)


if __name__ == '__main__':
    # we allow any case for the part in the dictionary, as python classes are
    # expected to be camel case
    data = [{'testcreatable': {'value': 4}},
            {'TestCreatable': {'value': 'hello creatable!'}}]

    for d in data:
        for obj, ctor in d.items():
            CreatableFactory.make(obj, ctor)

import yaml
from donkeycar.parts.part import CreatableFactory
from donkeycar import Vehicle
# -- !!! THIS import cannot be removed as otherwise the metaclass
# initialisation does not run for all parts !!!
import donkeycar.parts


class Builder:
    """
    Class that builds a donkey vehicle from a yaml file
    """
    def __init__(self, cfg, car_file='car.yml'):
        """
        :param cfg:         car config
        :param car_file:    car construction recipe
        """
        self.cfg = cfg
        self.car_file = car_file

    def insert_config(self, arguments):
        """
        Function to do in-place replacement of parameters called cfg.xyz or
        cfg.XYZ with XYZ from the car config file.
        :param dict arguments:  input/output dictionary
        """
        if arguments:
            for k, v in arguments.items():
                # go recursive if dictionary
                if type(v) is dict:
                    self.insert_config(v)
                if type(v) is str and v[:4].lower() == 'cfg.':
                    arguments[k] = getattr(self.cfg, v[4:])

    def build_vehicle(self):
        with open(self.car_file) as f:
            car_description = yaml.load(f, Loader=yaml.FullLoader)

        parts = car_description.get('parts')
        car = Vehicle()

        for part in parts:
            for part_name, part_params in part.items():
                # check if add_only_if is present
                if part_params.get('add_only_if') is False:
                    continue
                # we are using .get on part parameters here as the part might
                # not have it, then return an empty dict as ** needs to work
                part_args = part_params.get('arguments', {})
                # updated part parameters with config values
                self.insert_config(part_args)
                # this creates the part
                part = CreatableFactory.make(part_name, self.cfg, part_args)
                inputs = part_params.get('inputs', [])
                outputs = part_params.get('outputs', [])
                threaded = part_params.get('threaded', False)
                run_condition = part_params.get('run_condition')
                # adding part to vehicle
                car.add(part, inputs=inputs, outputs=outputs,
                        threaded=threaded, run_condition=run_condition)

        return car


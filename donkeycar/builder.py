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
                    overwrite = None
                    cfg = self.cfg
                    exec("overwrite = v")
                    arguments[k] = overwrite

    @staticmethod
    def insert_components(arguments, components):
        """
        Function to do in-place replacement of parameter string values that are
        objects, here called components. Components are defined in the
        components dictionary, where keys are the component names and values
        are dicts with types and arguments, eg:

        components:
          - input_pin_1:
              type: inputpin
              arguments:
                pin_id: RPI_GPIO.BCM.4

        :param dict arguments:      input/output dictionary
        :param dict components:     components dictionary
        """
        if arguments:
            for k, v in arguments.items():
                if type(v) is str and v in components:
                    component = components[v]
                    # create component as object
                    obj = CreatableFactory.make(component['type'],
                                                component['arguments'])
                    arguments[k] = obj

    def build_vehicle(self):
        with open(self.car_file) as f:
            car_description = yaml.load(f, Loader=yaml.FullLoader)

        parts = car_description.get('parts')
        car = Vehicle()

        for part in parts:
            for part_name, part_params in part.items():
                # replace any cfg parameters
                self.insert_config(part_params)
                # check if add_only_if is present
                if part_params.get('add_only_if') is False:
                    continue
                # we are using .get on part parameters here as the part might
                # not require any
                part_args = part_params.get('arguments')
                # updated part parameters with config values
                self.insert_config(part_args)
                # this creates the part
                vehicle_part = CreatableFactory.make(part_name, part_args)
                inputs = part_params.get('inputs', [])
                outputs = part_params.get('outputs', [])
                threaded = part_params.get('threaded', False)
                run_condition = part_params.get('run_condition')
                # adding part to vehicle
                car.add(vehicle_part, inputs=inputs, outputs=outputs,
                        threaded=threaded, run_condition=run_condition)

        return car


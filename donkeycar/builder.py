from copy import copy

import yaml
import graphviz
import logging
from donkeycar.parts.part import CreatableFactory
from donkeycar import Vehicle
# -- !!! THIS import cannot be removed as otherwise the metaclass
# initialisation does not run for all parts !!!
import donkeycar.parts

logger = logging.getLogger(__name__)


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
        """ This function creates the car from the yaml file"""
        with open(self.car_file) as f:
            try:
                car_description = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(e)
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

    def plot_vehicle(self, car):
        """ Function to provide a graphviz (dot) plot of the vehicle
            architecture. This is meant to be an auxiliary helper to control
            the data flow of input and output variables is as expected and
            input, output and run_condition variables are correctly labelled
            as they are free-form text strings. """
        g = graphviz.Digraph('Vehicle', filename='vehicle.py', format='png')
        g.attr('node', shape='box')

        # build all parts as nodes first
        car_parts = car.parts
        for part_dict in car_parts:
            part = part_dict['part']
            name = part.__class__.__name__
            label = name
            for k, v in part.kwargs.items():
                label += f'\n{k}: {str(v)}'
            g.node(name, label=label)

        # add all input, output, run_condition variables of all parts
        variable_tracker = set()
        for part_dict in car_parts:
            name = part_dict['part'].__class__.__name__
            variable_tracker.update((name, 'i', i) for i in part_dict.get('inputs', []))
            variable_tracker.update((name, 'o', o) for o in part_dict.get('outputs', []))
            run_condition = part_dict.get('run_condition')
            if run_condition:
                variable_tracker.add((name, 'r', run_condition))

        reversed_parts = list(reversed(car_parts))
        # connect all parts' outputs to inputs from top to bottom
        self.traverse_parts(car_parts, g, variable_tracker, 'blue')
        # reverse order and connect from bottom to top
        self.traverse_parts(reversed_parts, g, variable_tracker, 'green')
        # connected nodes now have all been removed, add the unconnected
        for node in variable_tracker:
            name, io, var = node
            # no input defined for input or run_condition
            if io in 'ir':
                g.edge('Non defined', name, var, color='red')
            # output goes nowhere
            if io == 'o':
                g.edge(name, 'Not used', var, color='red')
            print(node)
        g.view()

    @staticmethod
    def traverse_parts(car_parts, g, nodes, color='blue'):
        car_parts_copy = copy(car_parts)
        for part_dict in car_parts:
            outputs_upper = part_dict.get('outputs', [])
            upper_part_name = part_dict['part'].__class__.__name__
            print(f'Checking upper part {upper_part_name}')
            # copy contains all parts that come after the current part
            car_parts_copy.pop(0)
            for output_upper in outputs_upper:
                # here we cycle through all later parts
                for part_dict_lower in car_parts_copy:
                    inputs_lower = part_dict_lower.get('inputs', [])
                    lower_part_name = part_dict_lower['part'].__class__.__name__
                    print(f'Checking lower part {lower_part_name}')
                    if output_upper in inputs_lower:
                        g.edge(upper_part_name, lower_part_name,
                               label=output_upper, color=color)
                        found_output_upper = True
                        # remove found entries from all nodes
                        nodes.discard((upper_part_name, 'o', output_upper))
                        nodes.discard((lower_part_name, 'i', output_upper))

                    run_condition_lower = part_dict_lower.get('run_condition')
                    # if output is a run condition draw different edge
                    if run_condition_lower == output_upper:
                        g.edge(upper_part_name, lower_part_name,
                               label=output_upper, style='dotted',
                               color=color)
                        found_output_upper = True
                        nodes.discard((upper_part_name, 'o', output_upper))
                        nodes.discard((lower_part_name, 'r', output_upper))


if __name__ == "main":
    import donkeycar as dk
    from os.path import expanduser

    cfg = dk.load_config()
    yml = expanduser('~/Python/donkeycar/donkeycar/templates/vehicle_recipes'
                     '/test_vehicle.yml')
    b = Builder(cfg, yml)
    v = b.build_vehicle()
    # v.start()

    b.plot_vehicle(v)


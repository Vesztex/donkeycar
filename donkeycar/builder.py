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
                if part_params.get('add_only_if') is None or True:
                    # We are not using .get on part parameters here because the
                    # part might not have it. Rather return an empty dict
                    # because ** needs to work later on this return value
                    part_args = part_params.get('arguments', {})
                    # updated part parameters with config values
                    self.insert_config(part_args)
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
        """
        Function to provide a graphviz (dot) plot of the vehicle
        architecture. This is meant to be an auxiliary helper to control the
        data flow of input and output variables is as expected and input,
        output and run_condition variables are correctly labelled as they are
        free-form text strings.

        :param Vehicle car: input car to be plotted
        """
        g = graphviz.Digraph('Vehicle', filename='vehicle.py')
        g.attr('var_data', shape='box')

        # build all parts as nodes first
        car_parts = car.parts
        for part_dict in car_parts:
            part = part_dict['part']
            name = part.__class__.__name__
            label = name
            for k, v in part.kwargs.items():
                label += f'\n{k}: {str(v)}'
            label += f'\nThreaded: {"thread" in part_dict}'
            g.node(name, label=label)

        # add all input, output, run_condition variables of all parts
        variable_tracker = set()
        for part_dict in car_parts:
            name = part_dict['part'].__class__.__name__
            variable_tracker.update((name, 'i', i)
                                    for i in part_dict.get('inputs', []))
            variable_tracker.update((name, 'o', o)
                                    for o in part_dict.get('outputs', []))
            run_condition = part_dict.get('run_condition')
            if run_condition:
                variable_tracker.add((name, 'r', run_condition))

        reversed_parts = list(reversed(car_parts))
        # connect all parts' outputs to inputs from top to bottom
        self.traverse_parts(car_parts, g, variable_tracker, 'blue')
        # reverse order and connect from bottom to top
        self.traverse_parts(reversed_parts, g, variable_tracker, 'green')
        # connected nodes now have all been removed, add the unconnected
        for var_data in variable_tracker:
            name, io, var = var_data
            # no input defined for input or run_condition
            if io == 'i':
                g.edge('Non defined', name, var, color='red')
            elif io == 'r':
                g.edge('Non defined', name, var, color='red', style='dotted')
            # output goes nowhere
            elif io == 'o':
                g.edge(name, 'Not used', var, color='red')
            print(var_data)
        g.view()

    @staticmethod
    def traverse_parts(car_parts, g, var_data, color='blue'):
        """
        Method to traverse the parts list and edges to the nodes from outputs of
        higher parts to inputs and run_condition of lower parts. Output ->
        input edges are solid and output -> run_condition edges are dotted.

        The method also updates the passed in set of (parts, type, variable)
        tuples. For every found edge the corresponding start and end points
        are removed. Hence if the set contains all points it will contain
        only non-joined points after return.


        :param list car_parts:      list of parts as taken from the vehicle
        :param graphviz.Digraph g:  dot graph to be updated
        :param set var_data:        set of all variable data that gets updated
        :param string color:        color of edge
        :return: None
        """

        car_parts_copy = copy(car_parts)
        for part_dict in car_parts:
            outputs_upper = part_dict.get('outputs', [])
            upper_part_name = part_dict['part'].__class__.__name__
            # copy contains all parts that come after the current part
            car_parts_copy.pop(0)
            for output_upper in outputs_upper:
                # here we cycle through all later parts
                for part_dict_lower in car_parts_copy:
                    inputs_lower = part_dict_lower.get('inputs', [])
                    lower_part_name = part_dict_lower['part'].__class__.__name__
                    if output_upper in inputs_lower:
                        g.edge(upper_part_name, lower_part_name,
                               label=output_upper, color=color)
                        # remove found entries from all nodes
                        var_data.discard((upper_part_name, 'o', output_upper))
                        var_data.discard((lower_part_name, 'i', output_upper))

                    run_condition_lower = part_dict_lower.get('run_condition')
                    # if output is a run condition draw different edge
                    if run_condition_lower == output_upper:
                        g.edge(upper_part_name, lower_part_name,
                               label=output_upper, style='dotted',
                               color=color)
                        var_data.discard((upper_part_name, 'o', output_upper))
                        var_data.discard((lower_part_name, 'r', output_upper))


if __name__ == "__main__":
    import donkeycar as dk
    import os
    yml = os.path.join(os.path.dirname(dk.__file__), 'templates',
                       'vehicle_recipes', 'test_vehicle.yml')
    cfg = donkeycar.load_config(os.path.join(os.getcwd(), 'config.py'))
    b = Builder(cfg, yml)
    v = b.build_vehicle()
    b.plot_vehicle(v)
    # v.start()
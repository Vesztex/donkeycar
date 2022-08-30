import os
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
        assert os.path.exists(self.car_file), f"file {car_file} does not exist"

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

    def build_vehicle(self, kwargs={}):
        """
        This function creates the car from the yaml file. Allows overwriting
        of constructor parameters in the yaml through kwargs dictionary. For
        example, the yaml entry:
            - keraspilot:
                arguments:
                    model_type: linear

        can be alternatively passed by just having
            - keraspilot
        in the yaml file and passing {'keraspilot': 'linear'} in the kwargs
        argument. Data passed in the kwargs argument will overwrite data in the
        yaml file.

        Additionally, the builder supports passing parts as arguments for other
        parts. Hence, in a yaml file like
            - part_1
            - part_2:
                arguments:
                    part: part_1.0
        the builder will replace the string argument 'part_1.0' of the part_2
        arguments dictionary with the object that was created by the previous
        entry part_1. Note, that an increasing number '.0', '.1', '.2', etc is
        added to the part in order to identify different parts of the same type.
        Therefore, the above yaml corresponds to the following python code:
            p1 = Part_1()
            p2 = Part_2(part=p1)
        And if we have multiple versions of a part we can uniquely reference
        each of them like:
            - part_1
            - part_1
            - part_2:
                arguments:
                    part: part_1.0
        which translate into python code like:
            p1_0 = Part_1()
            p1_1 = Part_1()
            p2 = Part_2(part=p1_0)

        :param dict kwargs:     keyword arguments which will extend or
                                overwrite parameters in the yaml file
        :return Vehicle:        the assembled Vehicle
        """
        def extract(kwargs, part_name):
            d = dict()
            for k, v in kwargs.items():
                if k.startswith(f'{part_name}.') and v is not None:
                    d[k.replace(f'{part_name}.', '')] = v
            return d

        def update_with_previous_parts(part_args, car):
            """ Utility function to replace argument strings that refer to
                parts with the parts themselves. """
            for k, v in part_args.items():
                if type(v) is str:
                    prev_part = car.get_part_by_name(v)
                    if prev_part:
                        part_args[k] = prev_part

        def add_part(enable):
            # add part if either enable is missing or if it is present
            # and True or if it is a config parameter that is True,
            # passed like cfg.USE_PART for example
            if enable is None:
                return True
            if type(enable) is bool:
                return enable
            assert type(enable) is str, "Enable can only be None, bool or str"
            assert 'cfg.' in enable, "Enable must contain 'cfg.'"
            cfg = self.cfg
            return eval(enable)

        with open(self.car_file) as f:
            try:
                car_description = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(e)
        parts = car_description.get('parts')
        car = Vehicle()

        for part in parts:
            for part_name, part_params in part.items():
                enable = part_params.get('enable')
                if add_part(enable):
                    # We return an empty dict for missing arguments because **
                    # needs to work on this return value further down
                    args = part_params.get('arguments', {})
                    # update part parameters with config values
                    self.insert_config(args)
                    # update part parameters with kwargs
                    extract_kwargs = extract(kwargs, part_name)
                    args.update(extract_kwargs)
                    # update part parameters with already created parts
                    update_with_previous_parts(args, car)
                    # create part
                    part = CreatableFactory.make(part_name, self.cfg, **args)
                    # adding part to vehicle
                    inputs = part_params.get('inputs', [])
                    outputs = part_params.get('outputs', [])
                    threaded = part_params.get('threaded', False)
                    run_condition = part_params.get('run_condition')
                    car.add(part, inputs=inputs, outputs=outputs,
                            threaded=threaded, run_condition=run_condition)

        return car

    def plot_vehicle(self, car, view=True):
        """
        Function to provide a graphviz (dot) plot of the vehicle
        architecture. This is meant to be an auxiliary helper to control the
        data flow of input and output variables is as expected and input,
        output and run_condition variables are correctly labelled as they are
        free-form text strings.

        :param Vehicle car: input car to be plotted
        :param bool view:   if plot should be viewed instantaneously
        """
        g = graphviz.Digraph('Vehicle')
        g.attr('node', shape='box', ordering='out')
        car_parts = car.parts

        # build all parts as nodes first
        for part_dict in car_parts:
            name = part_dict['name']
            label = name
            for k, v in part_dict['part'].kwargs.items():
                label += f'\n{k}: {v}'
            label += f'\nThreaded: {"thread" in part_dict}'
            g.node(name, label=label)

        # add all input, output, run_condition variables of all parts to a
        # tracker
        variable_tracker = set()
        for part_dict in car_parts:
            name = part_dict['name']
            variable_tracker.update((name, 'i', i)
                                    for i in part_dict.get('inputs', []))
            variable_tracker.update((name, 'o', o)
                                    for o in part_dict.get('outputs', []))
            run_condition = part_dict.get('run_condition')
            if run_condition:
                variable_tracker.add((name, 'r', run_condition))

        # connect all parts' outputs to inputs from top to bottom
        self.traverse_parts(car_parts, g, variable_tracker)
        # connected nodes now have all been removed, add the unconnected
        for var_data in variable_tracker:
            name, io, var = var_data
            # no variable defined for input or run_condition
            if io == 'i':
                g.edge('Non defined', name, var, color='red')
            elif io == 'r':
                g.edge('Non defined', name, var, color='red', style='dotted')
            # output goes nowhere
            elif io == 'o':
                g.edge(name, 'Not used', var, color='red')

        base_name = os.path.splitext(os.path.basename(self.car_file))[0]
        filename = os.path.join(self.cfg.ASSEMBLY_PATH, base_name)
        rendered_file = g.render(filename=filename, view=view, cleanup=True)
        logger.info(f'Wrote file {rendered_file}')

    @staticmethod
    def traverse_parts(car_parts, g, var_data):
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
        :return: None
        """

        car_parts_copy_lower = copy(car_parts)
        car_parts_copy_upper = []
        for part_dict in car_parts:
            outputs = part_dict.get('outputs', [])
            part_name = part_dict['name']
            # copy contains all parts that come after the current part
            car_parts_copy_upper.append(car_parts_copy_lower.pop(0))
            for output in outputs:
                # parts in the same car loop that run afterwards
                Builder.create_edges(car_parts_copy_lower, 'blue', True, g,
                                     output, part_name, var_data)
                # parts in the next car loop
                Builder.create_edges(car_parts_copy_upper, 'green', False, g,
                                     output, part_name, var_data)

    @staticmethod
    def create_edges(car_parts, color, constraint, g, output,
                     part_name, var_data):
        """
        Method to draw the edges of the graph
        :param list car_parts:      car parts list to be connected
        :param str color:           color of edges
        :param bool constraint:     if graph should be strictly order top-down
        :param g graphviz.Digraphg: graph to be used
        :param str output:          name of output variable to be connected
        :param str part_name:       vertex (i.e. part) name where output
                                    variable resides
        :param set var_data:        recorded set of all parts, inputs, outputs,
                                    run_conditions which will be updated here
        :return:                    None
        """
        for part_dict in car_parts:
            inputs = part_dict.get('inputs', [])
            outputs = part_dict.get('outputs', [])
            run_condition = part_dict.get('run_condition')
            this_part_name = part_dict['name']
            if output in inputs:
                g.edge(part_name, this_part_name, label=output, color=color,
                       constraint=str(constraint))
                # remove found entries from tracker
                var_data.discard((part_name, 'o', output))
                var_data.discard((this_part_name, 'i', output))
            # if output is a run condition draw different edge
            if run_condition == output:
                g.edge(part_name, this_part_name, label=output, style='dotted',
                       color=color, constraint=str(constraint))
                var_data.discard((part_name, 'o', output))
                var_data.discard((this_part_name, 'r', output))
            # if a lower part overwrites the higher part's output,
            # then the higher part's output won't flow anywhere deeper
            # and the loop over lower parts stops for that higher output
            if output in outputs:
                break


def main(args):
    import os
    import argparse
    import donkeycar as dk
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument('--yaml', default=None,
                        help='path to assembly yaml file')
    for part_name, part_arg_list in CreatableFactory.arg_registry.items():
        for arg_name in part_arg_list:
            parser.add_argument(f'--{part_name}.{arg_name}')
    parsed_args = parser.parse_args(args)
    kwargs = vars(parsed_args)
    yml = parsed_args.yaml \
        or os.path.join(os.path.dirname(dk.__file__), 'templates',
                        'assembly', 'test_vehicle.yml')
    cfg = donkeycar.load_config(os.path.join(os.getcwd(), 'config.py'))
    b = Builder(cfg, yml)
    v = b.build_vehicle(kwargs)
    b.plot_vehicle(v, 'app')
    #v.start()
    v.stop()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(args)

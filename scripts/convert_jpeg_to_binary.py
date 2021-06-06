#!/usr/bin/env python3
"""
Usage:
    convert_jpeg_to_binary.py --tub=<path> --output=<path>

Note:
    This script converts a tub with jpeg images to one with binary images.
"""

import os
import traceback
import numpy as np
from docopt import docopt
from PIL import Image
from progress.bar import IncrementalBar
from donkeycar.parts.tub_v2 import Tub


def convert_jpeg_to_binary(path, output_path):
    """
    Convert from old tubs to new one

    :param path:                input tub paths
    :param output_path:         new tub output path
    :return:                    None
    """
    empty_record = {'__empty__': True}
    input_tub = Tub(path)

    # add input and type for empty records recording if they aren't present
    inputs = input_tub.manifest.inputs
    types = input_tub.manifest.types
    if '__empty__' not in inputs:
        inputs += ['__empty__']
        types += ['boolean']
    output_tub = Tub(base_path=output_path, inputs=inputs, types=types,
                     timestamp=input_tub.timestamp, img_as_jpeg=False)
    output_tub.manifest.manifest_metadata = input_tub.manifest.manifest_metadata
    num_records = len(input_tub)
    bar = IncrementalBar('Converting', max=num_records)
    previous_index = None
    for record in input_tub:
        try:
            img_path = record['cam/image_array']
            full_path = os.path.join(input_tub.base_path, 'images', img_path)
            full_path = os.path.expanduser(full_path)
            img = Image.open(full_path)
            img_arr = np.asarray(img)
            record['cam/image_array'] = img_arr
            current_index = record['_index']
            # first record or they are continuous, just append
            if not previous_index or current_index == previous_index + 1:
                output_tub.write_record(record)
                previous_index = current_index
            # otherwise fill the gap with empty records
            else:
                # Skipping over previous record here because it has
                # already been written.
                previous_index += 1
                # Adding empty record nodes, and marking them deleted
                # until the next valid record.
                delete_list = []
                while previous_index < current_index:
                    idx = output_tub.manifest.current_index
                    output_tub.write_record(empty_record)
                    delete_list.append(idx)
                    previous_index += 1
                output_tub.delete_records(delete_list)
            bar.next()
        except Exception as exception:
            print(f'Ignoring image path {full_path}\n', exception)
            traceback.print_exc()
    # writing session id into manifest metadata
    output_tub.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    input_path = args["--tub"]
    output_path = args["--output"]
    convert_jpeg_to_binary(input_path, output_path)

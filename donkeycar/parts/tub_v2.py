import os
import time
from operator import itemgetter

import numpy as np
from PIL import Image
import logging

from donkeycar.parts.datastore_v2 import Manifest, ManifestIterator


logger = logging.getLogger(__name__)


class Tub(object):
    """
    A datastore to store sensor data in a key, value format. \n
    Accepts str, int, float, image_array, image, and array data types.
    """

    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000, read_only=False):
        self.base_path = base_path
        self.images_base_path = os.path.join(self.base_path, Tub.images())
        self.metadata = metadata
        self.manifest = Manifest(base_path, inputs=inputs, types=types,
                                 metadata=metadata, max_len=max_catalog_len,
                                 read_only=read_only)
        self.input_types = dict(zip(inputs, types))
        # Create images folder if necessary
        if not os.path.exists(self.images_base_path):
            os.makedirs(self.images_base_path, exist_ok=True)

    def write_record(self, record=None):
        """
        Can handle various data types including images.
        """
        contents = dict()
        is_overwrite = '_index' in record
        for key, value in record.items():
            if value is None:
                continue
            elif key not in self.input_types:
                continue
            else:
                input_type = self.input_types[key]
                if input_type == 'float':
                    # Handle np.float() types gracefully
                    contents[key] = float(value)
                elif input_type == 'str':
                    contents[key] = value
                elif input_type == 'int':
                    contents[key] = int(value)
                elif input_type == 'boolean':
                    contents[key] = bool(value)
                elif input_type == 'nparray':
                    contents[key] = value.tolist()
                elif input_type == 'list' or input_type == 'vector':
                    contents[key] = list(value)
                elif input_type == 'image_array':
                    # Handle image array
                    image = Image.fromarray(np.uint8(value))
                    name = Tub._image_file_name(self.manifest.current_index, key)
                    image_path = os.path.join(self.images_base_path, name)
                    image.save(image_path)
                    contents[key] = name
        # Private properties, allow record overwriting if '_index' is given
        if is_overwrite:
            index = record['_index']
            contents['_timestamp_ms'] = record['_timestamp_ms']
            contents['_index'] = index
            # session id is optional and not necessary in all tubs
            session_id = record.get('_session_id')
            if session_id:
                contents['_session_id'] = session_id
        else:
            index = None
            # if timestamp already given in input dict then use it, otherwise
            # create from current time
            ts = record.get('_timestamp_ms') or int(round(time.time() * 1000))
            contents['_timestamp_ms'] = ts
            contents['_index'] = self.manifest.current_index
            contents['_session_id'] = self.manifest.session_id[1]

        self.manifest.write_record(contents, index)

    def delete_records(self, record_indexes):
        self.manifest.delete_records(record_indexes)

    def delete_last_n_records(self, n):
        # build ordered list of non-deleted indexes
        all_alive_indexes = sorted(set(range(self.manifest.current_index))
                                   - self.manifest.deleted_indexes)
        to_delete_indexes = all_alive_indexes[-n:]
        self.manifest.delete_records(to_delete_indexes)

    def restore_records(self, record_indexes):
        self.manifest.restore_records(record_indexes)

    def write_lap_times(self, overwrite=True):
        session_id = None
        lap = 0
        dist = 0
        time_stamp_ms = None
        lap_times = []
        res = {}
        # self is iterable
        for record in self:
            this_session_id = record.get('_session_id')
            if this_session_id != session_id:
                if session_id:
                    res[session_id] = lap_times
                    lap_times.clear()
                session_id = this_session_id
                time_stamp_ms = record['_timestamp_ms']
                dist = record['car/distance']
            this_lap = record['car/lap']
            if this_lap != lap:
                this_time_stamp_ms = record['_timestamp_ms']
                lap_time = (this_time_stamp_ms - time_stamp_ms) / 1000
                this_dist = record['car/distance']
                lap_dist = this_dist - dist
                lap_times.append(dict(lap=lap, time=lap_time, distance=lap_dist))
                lap = this_lap
                time_stamp_ms = this_time_stamp_ms
                dist = this_dist
        # add last session id
        res[session_id] = lap_times
        for sess_id, v in res:
            meta_session_id_dict = self.manifest.metadata.get(sess_id)
            if not meta_session_id_dict:
                self.manifest.metadata[sess_id] = dict(laptimer=v)
            elif 'laptimer' in meta_session_id_dict and overwrite or \
                    'laptimer' not in meta_session_id_dict:
                meta_session_id_dict['laptimer'] = v
        logger.info(f'Generated lap times {res}')

    def calculate_lap_performance(self, config):
        """
        Creates a dictionary of (session_id, lap) keys and int values
        where 0 is the fastest loop and num_bins-1 is the slowest.
        :param config:  donkey config to look up lap time pct bins
        :return:        dictionary of type ((session_id, lap, state_vector)
        """
        sessions \
            = self.manifest.manifest_metadata['sessions']['all_full_ids']
        session_lap_bin = {}
        for session_id in sessions:
            session_dict = self.manifest.metadata.get(session_id)
            assert session_dict, f"Missing metadata for session_id {session_id}"
            lap_timer = session_dict.get('laptimer')
            assert lap_timer, f"Missing laptimer in session_id {session_id} " \
                              f"metadata"
            # lap_timer is a list of dictionaries, sort here by time
            laps_sorted = sorted(lap_timer, key=itemgetter('time'))
            count = 0
            for i, lap_i in enumerate(laps_sorted):
                # jump over lap 0 as it is incomplete
                if lap_i == 0:
                    continue
                rel_i = i / len(laps_sorted)
                if rel_i > config.LAP_BINS[count]:
                    count += 1
                session_lap_bin[(session_id, lap_i['lap'])] \
                    = config.LAP_BINS[count]
        return session_lap_bin

    def close(self):
        logger.info(f'Closing tub {self.base_path}')
        self.manifest.close()

    def __iter__(self):
        return ManifestIterator(self.manifest)

    def __len__(self):
        return self.manifest.__len__()

    @classmethod
    def images(cls):
        return 'images'

    @classmethod
    def _image_file_name(cls, index, key, extension='.jpg'):
        key_prefix = key.replace('/', '_')
        name = '_'.join([str(index), key_prefix, extension])
        # Return relative paths to maintain portability
        return name


class TubWriter(object):
    """
    A Donkey part, which can write records to the datastore.
    """
    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000, lap_timer=None):
        self.tub = Tub(base_path, inputs, types, metadata, max_catalog_len)
        self.lap_timer = lap_timer

    def run(self, *args):
        assert len(self.tub.manifest.inputs) == len(args), \
            f'Expected {len(self.tub.manifest.inputs)} inputs but received' \
            f' {len(args)}'
        record = dict(zip(self.tub.manifest.inputs, args))
        self.tub.write_record(record)
        return self.tub.manifest.current_index

    def __iter__(self):
        return self.tub.__iter__()

    def close(self):
        # insert lap times into metadata of tub before closing
        if self.lap_timer:
            self.tub.manifest.metadata[self.tub.manifest.session_id] \
                = dict(laptimer=self.lap_timer.to_list())
        self.tub.close()

    def shutdown(self):
        self.close()


class TubWiper:
    """
    Donkey part which deletes a bunch of records from the end of tub.
    This allows to remove bad data already during recording. As this gets called
    in the vehicle loop the deletion runs only once in each continuous
    activation. A new execution requires to release of the input trigger. The
    action could result in a multiple number of executions otherwise.
    """
    def __init__(self, tub, num_records=20, min_loops=4):
        """
        :param tub: tub to operate on
        :param num_records: number or records to delete
        """
        self._tub = tub
        self._num_records = num_records
        self._active_loop_count = 0  # for debouncing
        self._min_loops = min_loops

    def run(self, is_delete):
        """
        Method in the vehicle loop. Delete records when trigger switches from
        False to True only.
        :param is_delete:   if deletion has been triggered by the caller
        :return:            true if triggered
        """
        # only run if input is true and debounced
        if is_delete:
            # increase the active loop count
            self._active_loop_count += 1
            # only trigger if we hit the counter, keeping the button pressed
            # will not trigger again, only increase the counter.
            if self._active_loop_count == self._min_loops:
                # action command
                self._tub.delete_last_n_records(self._num_records)
                # only trigger if it was released before
                logger.debug(f"Wiper triggered")
                return True
        else:
            # trigger released, reset active loop count
            self._active_loop_count = 0
        return False


from collections import defaultdict
from operator import itemgetter
import logging
from copy import copy

from donkeycar.parts.tub_v2 import Tub

logger = logging.getLogger(__name__)


class TubStatistics(object):
    """
    A datastore to store sensor data in a key, value format. \n
    Accepts str, int, float, image_array, image, and array data types.
    """

    def __init__(self, tub: Tub):
        self.tub = tub

    def generate_laptimes_from_records(self, overwrite=False):
        session_id = None
        lap = 0
        dist = 0
        time_stamp_ms = None
        lap_times = []
        res = {}
        # self is iterable
        for record in self.tub:
            this_session_id = record.get('_session_id')
            this_lap = record['car/lap']
            if this_session_id != session_id:
                # stepping into new session
                if session_id:
                    # copy results of current session
                    res[session_id] = copy(lap_times)
                    # reset lap_times and lap
                    lap_times.clear()
                lap = this_lap
                session_id = this_session_id
                time_stamp_ms = record['_timestamp_ms']
                dist = record['car/distance']

            if this_lap != lap:
                assert this_lap > lap, f'Found smaller lap {this_lap} than ' \
                                       f'previous lap {lap} in session {session_id}'
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
        for sess_id, lap_times in res.items():
            meta_session_id_dict = self.tub.manifest.metadata.get(sess_id)
            if not meta_session_id_dict:
                self.tub.manifest.metadata[sess_id] = dict(laptimer=lap_times)
            elif 'laptimer' in meta_session_id_dict and overwrite or \
                    'laptimer' not in meta_session_id_dict:
                meta_session_id_dict['laptimer'] = lap_times
        self.tub.manifest.write_metadata()
        logger.info(f'Generated lap times {res}')

    def calculate_lap_performance(self, use_lap_0=False, num_buckets=None):
        """
        Creates a dictionary of (session_id, lap) keys and int values
        where 0 is the fastest lap and num_bins-1 is the slowest.

        :param use_lap_0:   If the 0'th lap should be ignored. On the
                            real car lap zero shows up when the line is
                            crossed the first time hence the lap is
                            incomplete, but in the sim 0 indicates the
                            first complete lap
        :param num_buckets: If given, buckets the laps into as many buckets
                            and assigns the numbers i/num_buckets, i=1,...,
                            num_buckets to each lap in that bucket.
        :return:            dict of type {(session_id, lap): state_vector}
        """

        sessions \
            = self.tub.manifest.manifest_metadata['sessions']['all_full_ids']
        session_lap_rank = defaultdict(dict)
        logger.info(f'Calculating lap performance in tub {self.tub.base_path}')
        for session_id in sessions:
            session_dict = self.tub.manifest.metadata.get(session_id)
            assert session_dict, f"Missing metadata for session_id {session_id}"
            lap_timer = session_dict.get('laptimer')
            if not lap_timer:
                logger.warning(f"Missing or empty laptimer in session_id"
                               f" {session_id} metadata, skipping this id")
                continue
            # Remove lap zero if it shouldn't be considered. It should be first
            # entry, but check before removal.
            if not use_lap_0 and lap_timer[0]['lap'] == 0:
                del(lap_timer[0])
            # Remove laps that are not valid
            laps_filtered = [l for l in lap_timer if l.get('valid', True)]
            # lap_timer is a list of dictionaries, sort first by time and add
            # the relative timing into first entry of session_lap_rank and
            # then sort by distance and add this into the second entry
            for sort_by in ('time', 'distance'):
                laps_sorted = sorted(laps_filtered, key=itemgetter(sort_by))
                num_laps = len(laps_sorted)
                for i, lap_i in enumerate(laps_sorted):
                    if num_buckets is None:
                        rel_i = (i + 1) / num_laps
                    else:
                        rel_i = int(i * num_buckets/num_laps + 1) / num_buckets
                    # put time sorting into first and distance sorting into
                    # second entry of tuple
                    if sort_by == 'time':
                        # when we loop over time, set a list with second
                        # entry being empty to be filled with distance ranks
                        session_lap_rank[session_id][lap_i['lap']] \
                            = [rel_i, None]
                    else:
                        # when we loop over distance, fill the empty entry
                        session_lap_rank[session_id][lap_i['lap']][1] = rel_i
                log_text = f'Session {session_id} with {num_laps} valid laps ' \
                           f'out of {len(lap_timer)}'
                if num_laps > 0:
                    log_text \
                        += f', min {sort_by}: {laps_sorted[0][sort_by]:5.2f}, '\
                           f'max {sort_by}: {laps_sorted[-1][sort_by]:5.2f}'
                logger.info(log_text)
        return session_lap_rank

    def all_lap_times(self):
        """ returns {session_id_1: { lap_i: time_i, ...}, session_id_2:... } """
        d = {s_id: {lap_timer_i['lap']: lap_timer_i['time'] for
                    lap_timer_i in v['laptimer']} for s_id, v in
             self.tub.manifest.metadata.items()}
        return d

    def calculate_aggregated_gyro(self, use_lap_0=False, session_lap_rank=None):
        """
        Creates a dictionary of (session_id, lap) keys and float values of
        aggregated gyro_z values. Note, in the sim, thy gyro z is in the
        middle coordinate, not the last.

        :param use_lap_0:           If the 0'th lap should be ignored. On the
                                    real car lap zero shows up when the line is
                                    crossed the first time hence the lap is
                                    incomplete, but in the sim 0 indicates the
                                    first complete lap,
        :param session_lap_rank:    Optional session / lap dictionary. This will
                                    be updated instead of producing a new
                                    dictionary if given.

        :return:                    dict of type {session_id: {lap: float}}

        """
        def calculate_gyro_pct(session_lap_gyro, this_session):
            # normalise the gyro_z values by dividing by max value
            gyro_data = session_lap_gyro[this_session]
            values = gyro_data.values()
            if not values:
                logger.info(f'Skipping normalising of gyro data for session '
                            f'{this_session}, because it is empty.')
                return
            logger.info(f'Normalising gyro data for session {this_session}')
            gyro_max = max(values)
            gyro_min = min(values)
            for lap_i, gyro_z in gyro_data.items():
                # because dicts are shallow copies this updates in-place
                gyro_data[lap_i] = (gyro_z - gyro_min) / (gyro_max - gyro_min)

        session_lap_gyro = dict()
        logger.info(f'Calculating aggregated gyro in tub {self.tub.base_path}')
        prev_session = None
        laps_filtered = []
        # this loop assumes that all records for one session are continuous,
        # i.e. if we iterate over all records then if a change to a new
        # session happens only once for each session, see assert below
        for record in self.tub:
            # skip over zero lap
            lap = record['car/lap']
            if not use_lap_0 and lap == 0:
                continue

            session_id = record['_session_id']
            if prev_session != session_id:
                # If new session found assert we don't have this session yet.
                assert session_id not in session_lap_gyro, \
                    f'Session {session_id} already found, unordered tub ' \
                    f'detected'

                session_dict = self.tub.manifest.metadata.get(session_id)
                assert session_dict, \
                    f"Missing metadata for session_id {session_id}"
                lap_timer = session_dict.get('laptimer')
                if not lap_timer:
                    logger.warning(f"Missing or empty laptimer in session_id"
                                   f" {session_id} metadata, skipping this id")
                    continue
                laps_filtered = [l['lap'] for l in lap_timer
                                 if l.get('valid', True)]
                logger.info(f'Finding {len(laps_filtered)} valid laps out of '
                            f'{len(lap_timer)} in session {session_id}')
                # Create a default dict entry for the session.
                session_lap_gyro[session_id] = defaultdict(float)
                if prev_session:
                    # Only calculate the stats of the prev session at the
                    # beginning of a new session
                    calculate_gyro_pct(session_lap_gyro, prev_session)
                # update current session to new session
                prev_session = session_id

            # jump over records of non-complete laps
            if lap not in laps_filtered:
                continue
            # take the second coordinate here, this is gyro_z in sim
            val = abs(record['car/gyro'][1])
            session_lap_gyro[session_id][lap] += val

        # Calculate the stats of the last
        calculate_gyro_pct(session_lap_gyro, prev_session)

        if not session_lap_rank:
            return session_lap_gyro

        # If session_lap_rank given, then we assume this data contains the
        # percentage rankings for time and distance already, and we merge the
        # gyro ranking into it
        for session, lap_value in session_lap_rank.items():
            for lap_i, value_i in lap_value.items():
                # assert lap_value is a list of length 2
                assert isinstance(value_i, list) and len(value_i) == 2,\
                    f'Expect the entry of session_lap_dict[{session}][{lap_i}]'\
                    f' to be a 2d list'
                gyro_value = session_lap_gyro[session][lap_i]
                value_i.append(gyro_value)

        return session_lap_rank

from collections import defaultdict
from operator import itemgetter
import logging
from copy import copy

from donkeycar.parts.tub_v2 import Tub

logger = logging.getLogger(__name__)


class TubStatistics(object):
    """
    A statistics calculator for tub data. Sorts and calculates quantiles for
    lap times, distances and gyro data.
    """

    def __init__(self, tub: Tub, gyro_z_index: int = 1):
        """
        Construct tub statistics calculator for tub

        :param tub:             input tub
        :param gyro_z_index:    z coordinate in 3d gyro vector (this is 1 in
                                sim)
        """
        self.tub = tub
        self.gyro_z_index = gyro_z_index

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

        self._calculate_aggregated_gyro()
        sessions \
            = self.tub.manifest.manifest_metadata['sessions']['all_full_ids']
        session_lap_rank = defaultdict(lambda: defaultdict(dict))
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
            # lap_timer is a list of dictionaries, sort by time, distance and
            # gyro_z aggregated respectively and record the quantile bins for
            # each of these values per lap in the session_lap_rank dict.
            num_laps = len(laps_filtered)
            for sort_by in ('time', 'distance', 'gyro_z_agg'):
                laps_sorted = sorted(laps_filtered, key=itemgetter(sort_by))
                for i, lap_i in enumerate(laps_sorted):
                    if num_buckets is None:
                        num_buckets = num_laps
                    rel_i = int(i * num_buckets/num_laps + 1) / num_buckets
                    session_lap_rank[session_id][lap_i['lap']][sort_by] = rel_i
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

    def _calculate_aggregated_gyro(self):
        """
        Updates lap_timer in tub metadate with values of aggregated gyro_z
        values. Note, in the sim, thy gyro z is in the middle coordinate,
        not the last.
        """

        logger.info(f'Calculating aggregated gyro in tub {self.tub.base_path}')
        prev_session = None
        prev_lap = None
        gyro_z_agg = 0
        lap_gyro_map = {}
        # this loop assumes that all records for one session are continuous,
        # i.e. if we iterate over all records then if a change to a new
        # session happens only once for each session, see assert below
        for record in self.tub:
            # skip over zero lap
            lap = record['car/lap']
            session_id = record['_session_id']

            if session_id != prev_session:
                # If new session found update the lap timer with the
                # aggregated values
                if prev_session is not None:

                    session_dict = self.tub.manifest.metadata.get(prev_session)
                    assert session_dict, \
                        f"Missing metadata for session_id {prev_session}"
                    lap_timer = session_dict.get('laptimer')
                    # update lap_timer here
                    for entry in lap_timer:
                        lap_i = entry['lap']
                        entry['gyro_z_agg'] = lap_gyro_map[lap_i]

                # update current session to new session
                prev_session = session_id

                # reset lap / gyro map
                lap_gyro_map.clear()
                prev_lap = None

            if lap != prev_lap:
                if prev_lap is not None:
                    # add aggregated gyro value to map
                    lap_gyro_map[prev_lap] = gyro_z_agg
                    # zero the aggregation value
                    gyro_z_agg = 0
                # update the lap
                prev_lap = lap

            val = abs(record['car/gyro'][self.gyro_z_index])
            gyro_z_agg += val

        # write back
        self.tub.manifest.write_metadata()



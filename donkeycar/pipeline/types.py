import copy
import os
from typing import Any, List, Optional, Iterator, Union, Iterable
import logging
import numpy as np
from donkeycar.config import Config
from donkeycar.parts.tub_statistics import TubStatistics
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import load_image, load_pil_image
from typing_extensions import TypedDict


logger = logging.getLogger(__name__)

TubRecordDict = TypedDict(
    'TubRecordDict',
    {
        '_index': int,
        '_session_id': str,
        'cam/image_array': str,
        'user/angle': float,
        'user/throttle': float,
        'user/mode': str,
        'imu/acl_x': Optional[float],
        'imu/acl_y': Optional[float],
        'imu/acl_z': Optional[float],
        'imu/gyr_x': Optional[float],
        'imu/gyr_y': Optional[float],
        'imu/gyr_z': Optional[float],
        'behavior/one_hot_state_array': Optional[List[float]],
        'localizer/location': Optional[int],
        'car/accel': Optional[List[float]],
        'car/gyro': Optional[List[float]],
        'car/speed': Optional[float],
        'car/lap': Optional[int],
        'lap_pct': Optional[float]
    }
)


# def softmax_plus(n):
#     logn = max(math.log(n), 1)
#     return [(1 / (i+1)) * logn for i in range(n)]


def softmax_plus(n):
    x = np.arange(n) * 20 / n
    return n * np.exp(-x) / np.exp(-x).sum()


class TubRecord(object):
    def __init__(self, config: Config, base_path: str,
                 underlying: TubRecordDict) -> None:
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._cache_images = getattr(self.config, 'CACHE_IMAGES', True)
        self._image: Optional[Any] = None

    def image(self, processor=None, as_nparray=True) -> np.ndarray:
        """
        Loads the image.

        :param processor:   Image processing like augmentations or cropping, if
                            not None. Defaults to None.
        :param as_nparray:  Whether to convert the image to a np array of uint8.
                            Defaults to True. If false, returns result of
                            Image.open()
        :return:            Image
        """
        if self._image is None:
            image_path = self.underlying['cam/image_array']
            full_path = os.path.join(self.base_path, 'images', image_path)

            if as_nparray:
                _image = load_image(full_path, cfg=self.config)
            else:
                # If you just want the raw Image
                _image = load_pil_image(full_path, cfg=self.config)
            if processor:
                _image = processor(_image)
            # only cache images if config does not forbid it
            if self._cache_images:
                self._image = _image
        else:
            _image = self._image
        return _image

    def extend(self, session_lap_rank):
        if not session_lap_rank:
            return True
        session_id = self.underlying['_session_id']
        lap_i = self.underlying['car/lap']

        if session_lap_rank:
            # we won't get a result for the last lap as this is incomplete and
            # doesn't have a time.
            lap_i_dict = session_lap_rank[session_id].get(lap_i)
            if lap_i_dict:
                lap_pct = [lap_i_dict[key] for key
                           in ('time', 'distance', 'gyro_z_agg')]
                self.underlying['lap_pct'] = lap_pct

        return lap_i_dict is not None

    def __repr__(self) -> str:
        return repr(self.underlying)


class TubDataset(object):
    """
    Loads the dataset and creates a TubRecord list (or list of lists).
    """

    def __init__(self, config: Config, tub_paths: List[str],
                 seq_size: int = 0, add_lap_pct: str = '') -> None:
        self.config = config
        self.tub_paths = tub_paths
        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = list()
        self.train_filter = getattr(config, 'TRAIN_FILTER', None)
        self.add_lap_pct = add_lap_pct.lower()
        self.seq_size = seq_size
        logger.info(f'Created TubDataset with add_lap_pct: {self.add_lap_pct}')

    def get_records(self) -> Union[List[TubRecord], List[List[TubRecord]]]:
        if not self.records:
            filtered_records = 0
            non_ext_records = 0
            used_records = 0
            logger.info(f'Loading tubs from paths {self.tub_paths}')
            for tub in self.tubs:
                tub_stat = TubStatistics(tub)
                session_lap_rank = tub_stat.calculate_lap_performance(
                    self.config.USE_LAP_0, num_buckets=4, compress=True)
                for underlying in tub:
                    record = TubRecord(self.config, tub.base_path, underlying)
                    if self.train_filter and not self.train_filter(record):
                        filtered_records += 1
                    elif record.extend(session_lap_rank):
                        self.records.append(record)
                        used_records += 1
                    else:
                        non_ext_records += 1
            total_records = used_records + filtered_records + non_ext_records
            logger.info(f'Records: # Total {total_records}  # Used '
                        f'{used_records}  # Filtered {filtered_records}  # '
                        f'NonExtended {non_ext_records}')
            if self.seq_size > 0:
                seq = Collator(self.seq_size, self.records)
                self.records = list(seq)
        return self.records

    @staticmethod
    def convert_to_weight(session_lap_rank):
        for session, lap_rank in session_lap_rank.items():
            # lap_rank is ordered dictionary of lap number vs lap pct,
            # we replace the value lap pct with a weight given by the order
            num_laps = len(lap_rank)
            weights = softmax_plus(num_laps)
            for i_weight, key in enumerate(lap_rank.keys()):
                lap_rank[key] = weights[i_weight]

    def close(self):
        logger.info(f'Closing TubDataset')
        for tub in self.tubs:
            tub.close()


class Collator(Iterable[List[TubRecord]]):
    """" Builds a sequence of continuous records for RNN and similar models. """
    def __init__(self, seq_length: int, records: List[TubRecord]):
        """
        :param seq_length:  length of sequence
        :param records:     input record list
        """
        self.records = records
        self.seq_length = seq_length

    @staticmethod
    def is_continuous(rec_1: TubRecord, rec_2: TubRecord) -> bool:
        """
        Checks if second record is next to first record

        :param rec_1:   first record
        :param rec_2:   second record
        :return:        if first record is followed by second record
        """
        it_is = rec_1.underlying['_index'] == rec_2.underlying['_index'] - 1 \
                and '__empty__' not in rec_1.underlying \
                and '__empty__' not in rec_2.underlying
        return it_is

    def __iter__(self) -> Iterator[List[TubRecord]]:
        """ Iterable interface. Returns a generator as Iterator. """
        it = iter(self.records)
        for this_record in it:
            seq = [this_record]
            seq_it = copy.copy(it)
            for next_record in seq_it:
                if self.is_continuous(this_record, next_record) and \
                        len(seq) < self.seq_length:
                    seq.append(next_record)
                    this_record = next_record
                else:
                    break
            if len(seq) == self.seq_length:
                yield seq



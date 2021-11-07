import copy
import os
from typing import Any, List, Optional, TypeVar, Iterator, Iterable, \
    Callable, Union
import logging
import numpy as np
from donkeycar.config import Config
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import load_image, load_pil_image
from typing_extensions import TypedDict
from PIL.Image import Image

logger = logging.getLogger(__name__)

X = TypeVar('X', covariant=True)

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


class TubRecord(object):
    def __init__(self, config: Config, base_path: str,
                 underlying: TubRecordDict) -> None:
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._image: Optional[Any] = None

    def image(self, cached: bool = None,
              as_nparray: bool = True,
              transformation:  Callable[[np.ndarray], np.ndarray] = None) \
            -> np.ndarray:
        """Loads the image for you
        Args:
            cached:         whether to cache the image. Defaults to True.
            as_nparray:     whether to convert the image to a np.array of
                            uint8. Defaults to True. If false, returns result
                            of Image.open()
            transformation: image transformation to be applied

        Returns: Image, either np array or Pil image
        """

        if self._image is None:
            image_path = self.underlying['cam/image_array']
            full_path = os.path.join(self.base_path, 'images', image_path)

            if as_nparray:
                _image = load_image(full_path, cfg=self.config)
                if transformation:
                    _image = transformation(_image)
            else:
                # If you just want the raw Image
                _image = load_pil_image(full_path, cfg=self.config)

            if cached is None:
                cached = self.config.CACHE_IMAGES
            if cached:
                self._image = _image
        else:
            _image = self._image
        return _image

    def extend(self, session_lap_bin):
        if not session_lap_bin:
            return True
        session_id = self.underlying['_session_id']
        lap_i = self.underlying['car/lap']
        # we won't get a result for the last lap as this is incomplete and
        # doesn't have a time.
        pct = session_lap_bin.get((session_id, lap_i))
        self.underlying['lap_pct'] = pct
        return pct is not None

    def __repr__(self) -> str:
        return repr(self.underlying)


class TubDataset(object):
    """
    Loads the dataset and creates a TubRecord list (or list of lists).
    """

    def __init__(self, config: Config, tub_paths: List[str],
                 seq_size: int = 0,
                 add_lap_pct: bool = False) -> None:
        self.config = config
        self.tub_paths = tub_paths
        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = list()
        self.train_filter = getattr(config, 'TRAIN_FILTER', None)
        self.add_lap_pct = add_lap_pct
        self.seq_size = seq_size

    def get_records(self) -> Union[List[TubRecord], List[List[TubRecord]]]:
        if not self.records:
            filtered_records = 0
            logger.info(f'Loading tubs from paths {self.tub_paths}')
            for tub in self.tubs:
                session_lap_bin = None
                if self.add_lap_pct:
                    session_lap_bin = tub.calculate_lap_performance(self.config)
                for underlying in tub:
                    record = TubRecord(self.config, tub.base_path, underlying)
                    if self.train_filter and not self.train_filter(record):
                        filtered_records += 1
                    elif record.extend(session_lap_bin):
                        self.records.append(record)
            logger.info(f'Filtered out {filtered_records} records')
            if self.seq_size > 0:
                seq = Collator(self.seq_size, self.records)
                self.records = list(seq)
        return self.records

    def close(self):
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



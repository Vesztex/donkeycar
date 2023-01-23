from typing import Callable, Iterator, List, Sized, Dict, Tuple, Iterable
from donkeycar.pipeline.types import TubRecord


class PipelineGenerator(Sized):
    def __init__(self,
                 records: Iterable[TubRecord],
                 x_transform: Callable[[TubRecord], Dict],
                 y_transform: Callable[[TubRecord], Dict],
                 w_transform: Callable[[TubRecord], Dict] = None) -> None:
        """
        Generator for TF data. Reads an iterable of tub records and provides
        a generator for iterating through the tub records and returning a
        tuple of (x, y) or (x, y, z) where x, y are the models inputs and
        outputs in dictionary form and z is an optional dictionary of
        weights. For consistency z needs to have the same keys TensorShapes
        as y.

        :param records:         Iterable of TubRecords
        :param x_transform:     Callable to extract x from TubRecord
        :param y_transform:     Callable to extract y from TubRecord
        :param w_transform:     Callable to extract weights from TubRecord
        """
        self.records = records
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.w_transform = w_transform

    def __iter__(self) -> Iterator[Tuple[Dict, Dict]]:
        for record in self.records:
            ret = self.x_transform(record), self.y_transform(record)
            if self.w_transform:
                ret = *ret, self.w_transform(record)
            yield ret

    def __len__(self) -> int:
        # this might fail if self.records is not a list (i.e. Sizable).
        return len(self.records)


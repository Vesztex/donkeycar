from typing import Callable, Iterator, List, Sized, Dict, Tuple
from donkeycar.pipeline.types import TubRecord


class PipelineGenerator(Sized):
    def __init__(self,
                 records: List[TubRecord],
                 x_transform: Callable[[TubRecord], Dict],
                 y_transform: Callable[[TubRecord], Dict]) -> None:
        self.records = records
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __iter__(self) -> Iterator[Tuple[Dict, Dict]]:
        for record in self.records:
            yield self.x_transform(record), self.y_transform(record)

    def __len__(self) -> int:
        return len(self.records)


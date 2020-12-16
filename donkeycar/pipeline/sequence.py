from typing import Any, Callable, Generic, Iterable, Iterator, List, Sized, \
    Tuple, TypeVar, Union

from donkeycar.pipeline.types import TubRecord

# Note: Be careful when re-using `TypeVar`s.
# If you are not-consistent mypy gets easily confused.

R = TypeVar('R', covariant=True)
X = TypeVar('X', covariant=True)
Y = TypeVar('Y', covariant=True)
XOut = TypeVar('XOut', covariant=True)
YOut = TypeVar('YOut', covariant=True)


# This is a protocol type without explicitly using a `Protocol`
# Using `Protocol` requires Python 3.7
class SizedIterable(Generic[X], Iterable[X], Sized):
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[X]:
        pass


class TubSeqIterator(Iterator[TubRecord]):
    def __init__(self, records: List[TubRecord]) -> None:
        self.records = records or list()
        self.iter = iter(self.records)

    def __iter__(self) -> 'TubSeqIterator[TubRecord]':
        return self

    def __next__(self) -> TubRecord:
        return next(self.iter)


class _BaseTfmIterator(Iterator[Tuple[XOut, YOut]]):
    """
    A basic transforming iterator. Do no use this class directly.
    """
    def __init__(self,
                 iterable: [Union[SizedIterable[R],
                                  SizedIterable[Tuple[X, Y]]]]) \
            -> None:
        self.iterable = iterable
        self.iterator = iter(self.iterable.iterable)

    def __iter__(self) -> '_BaseTfmIterator[XOut, YOut]':
        return self

    def __next__(self) -> Tuple[XOut, YOut]:
        r = next(self.iterator)
        if isinstance(r, tuple) and len(r) == 2:
            x, y = r
            return self.iterable.x_transform(x), self.iterable.y_transform(y)
        else:
            return self.iterable.x_transform(r), self.iterable.y_transform(r)


class TfmShallowList(Generic[R, XOut, YOut], SizedIterable[Tuple[XOut, YOut]]):
    def __init__(self,
                 iterable: Union['TubSequence',
                                 'TfmShallowList[X, Y, XOut, YOut]'],
                 x_transform: Callable[[Union[R, X]], XOut],
                 y_transform: Callable[[Union[R, Y]], YOut]) -> None:
        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return len(self.iterable)

    def __iter__(self) -> _BaseTfmIterator[XOut, YOut]:
        return _BaseTfmIterator(self)


class TubSequence(SizedIterable[TubRecord]):
    def __init__(self, records: List[TubRecord]) -> None:
        self.records = records

    def __iter__(self) -> TubSeqIterator:
        return TubSeqIterator(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def build_pipeline(self,
                       x_transform: Callable[[TubRecord], X],
                       y_transform: Callable[[TubRecord], Y]) \
            -> TfmShallowList[TubRecord, X, Y]:
        return TfmShallowList(self,
                              x_transform=x_transform,
                              y_transform=y_transform)

    @classmethod
    def map_pipeline(cls,
                     pipeline: Union['TubSequence',
                                     TfmShallowList[TubRecord, X, Y]],
                     x_transform: Callable[[X], XOut],
                     y_transform: Callable[[Y], YOut]) \
            -> TfmShallowList[Tuple[X, Y], XOut, YOut]:
        return TfmShallowList(pipeline,
                              x_transform=x_transform,
                              y_transform=y_transform)


class Pipeline(SizedIterable[Tuple[XOut, YOut]]):
    def __init__(self,
                 iterable: SizedIterable[Union[R, Tuple[X, Y]]],
                 x_transform: Callable[[Union[R, X]], XOut],
                 y_transform: Callable[[Union[R, Y]], YOut]) -> None:
        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __iter__(self) -> Iterator[Tuple[XOut, YOut]]:
        for record in self.iterable:
            if type(record) is tuple:
                x, y = record
                yield self.x_transform(x), self.y_transform(y)
            else:
                yield self.x_transform(record), self.y_transform(record)

    def __len__(self) -> int:
        return len(self.iterable)

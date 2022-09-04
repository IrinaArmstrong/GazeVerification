import numpy as np
import collections
from itertools import islice
from typing import Union, TypeVar, List

DataType = TypeVar("DataType")


def sliding_window(sequence: Union[list, tuple, np.ndarray],
                   window_size: int = 2,
                   step: int = 2,
                   fill_value: DataType = None) -> List[DataType]:
    """
    Performs a rolling window (aka sliding window) iterable over a sequence/iterator/generator.
    :param sequence: a sequence to iterate over,
    :type sequence: a sequence/iterator/generator.
    :param window_size: the size of a single data slice - window,
    :type window_size: int,
    :param step: the number of data points between beginnings of a pair of sequential windows,
    :type step: int,
    :param fill_value: value for filling missing parts of a window in some particular cases,
    :type fill_value: any
    :return: each time - a list of a sequence of length `window_size`,
    :rtype: list.
    """
    if not len(sequence):
        raise ValueError(f"Sequence provided for sliding window is empty!")
    if window_size < 0 or step < 1:
        raise ValueError(f"Sliding window function received inappropriate parameters configuration:"
                         " 'window_size' < 0 or 'step' < 0")
    if (fill_value is not None) and (type(fill_value) != type(sequence[0])):
        raise ValueError(f"Filling value type ({type(fill_value)}) do not match "
                         f"sequence elements type ({type(sequence[0])})")
    it = iter(sequence)
    q = collections.deque(islice(it, window_size), maxlen=window_size)
    if not q:
        return list()
    q.extend(fill_value for _ in range(window_size - len(q)))  # pad to window size
    while True:
        yield list(q)
        try:
            q.append(next(it))
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fill_value) for _ in range(step - 1))

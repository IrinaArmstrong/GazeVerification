import numpy as np
import collections
from copy import deepcopy
from itertools import islice
from typing import Union, TypeVar, List, Generator, Tuple

DataType = TypeVar("DataType")


def generative_sliding_window_1d(sequence: Union[list, tuple, np.ndarray],
                                 window_size: int = 2,
                                 step: int = 2,
                                 fill_value: DataType = None) -> Generator[DataType]:
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
    :rtype: generator.
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


def sequential_sliding_window_1d(data: Union[list, tuple, np.ndarray], window_size: Tuple[int, int],
                                 dx: int = 1) -> np.ndarray:
    """
    Generated a sequence of 1d sliding windows over input M-dim data.
    :param data: input data for windows generating,
    :type data: array-like types: list, tuple, np.array ...
    :param window_size: the sizes of sliding window,
    :type window_size: tuple,
    :param dx: a step over horizontal axis (x),
    :type dx: int,
    :return: array of size [1, num windows, window_size[0], window_size[1]]
    :rtype: array.

    Note: Stride operations in Numpy manipulates the internal data structure of ndarray and,
    if done incorrectly, the array elements can point to invalid memory
    and can corrupt results.
    So making a copy of input data for transformations.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    data_ = deepcopy(data)
    shape = data_.shape[:-1] + (int((data_.shape[-1] - window_size[-1]) / dx) + 1,) + window_size
    strides = data_.strides[:-1] + (data_.strides[-1] * dx,) + data_.strides[-1:]
    return np.lib.stride_tricks.as_strided(data_, shape=shape, strides=strides)


def sequential_sliding_window_2d(data: Union[list, tuple, np.ndarray], window_size: Tuple[int, int],
                                 dx: int = 1, dy: int = 1) -> np.ndarray:
    """
    Generated a sequence of 2d sliding windows over input M-dim data.
    :param data: input data for windows generating,
    :type data: array-like types: list, tuple, np.array ...
    :param window_size: the sizes of sliding window,
    :type window_size: tuple,
    :param dx: a step over horizontal axis (x),
    :type dx: int,
    :param dy: a step over vertical axis (y),
    :type dy: int,
    :return: array of size [1, num windows, window_size[0], window_size[1]]
    :rtype: array.

    Note: Stride operations in Numpy manipulates the internal data structure of ndarray and,
    if done incorrectly, the array elements can point to invalid memory
    and can corrupt results.
    So making a copy of input data for transformations.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    data_ = deepcopy(data)
    shape = data_.shape[:-2]  # other input tensor dimensions (i.e. batch size ...)
    shape += ((data_.shape[-2] - window_size[-2]) // dy + 1,)  # number of window steps on y axis
    shape += ((data_.shape[-1] - window_size[-1]) // dx + 1,)  # number of window steps on x axis
    shape += window_size

    # (strides[0] - on rows, strides[1] - on cols)
    strides = data_.strides[:-2]  # a step over other input tensor dimensions (i.e. batch size ...)
    strides += (data_.strides[-2] * dy,)  # a step over singular axis (of shape 1)
    strides += (data_.strides[-1] * dx,)  # a step between windows (= window_size[0] * window_size[1])
    strides += data_.strides[-2:]  # a step over windows

    return np.lib.stride_tricks.as_strided(data_, shape=shape, strides=strides)

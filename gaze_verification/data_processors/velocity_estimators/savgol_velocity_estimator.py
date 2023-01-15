import torch
import numpy as np
from math import factorial
from typeguard import typechecked
from typing import Union, Any, List, Optional

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.filters.filtration_utils import DerivativeOrder
from gaze_verification.data_processors.velocity_estimators.velocity_estimator_abstract import VelocityEstimatorAbstract


@typechecked
class SavitzkyGolayVelocityEstimator(VelocityEstimatorAbstract):
    """
    The Savitzky Golay filter is a particular type of low-pass filter, well adapted for data smoothing.
    Also Savitzky-Golay filters producing smoothened signal gradients / derivatives.
    The main point of Savitzky-Golay gradients is to compute the derivative at a given point,
    simply by computing the derivative of the fitted polynomial.

    For further information see:
        description: https://bartwronski.com/2021/11/03/study-of-smoothing-filters-savitzky-golay-filters/
        implementation details: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    References
    ----------
    .. [1] Eye Know You Too: A DenseNet Architecture for End-to-end Biometric Authentication via Eye Movements.
        (arXiv:2201.02110v1 [cs.CV])  01/05/2022 by  Dillon Lohr, et al. Texas State University.
    """

    def __init__(self,
                 window_size: int,
                 order: int,
                 rate: Optional[int] = 1,
                 keep_erroneous_samples: bool = True,
                 verbose: bool = True):
        super().__init__(verbose)
        self.window_size = window_size
        self.order = order
        self.rate = rate
        self.derivative = DerivativeOrder.FIRST_ORDER  # 1 - for first order derivative - velocity
        self.keep_erroneous_samples = keep_erroneous_samples
        self.check_selected_parameters()
        self._hyperparameters = self.register_hyperparameters()

    def check_selected_parameters(self):
        """
        Checks whether the selected window size and order pair is valid:
         - Window size is required to be a positive odd number, larger then 1,
         - Window size is required to be larger then selected order at least on 2
            (1 point for each window side),
        """
        try:
            self.window_size = np.abs(int(self.window_size))
            self.order = np.abs(int(self.order))
        except ValueError as e:
            raise ValueError("Window size and order have to be of type 'int'")
        if self.window_size % 2 != 1 or self.window_size <= 1:
            raise TypeError("Window size must be a positive odd number larger then 1")
        if self.window_size < self.order + 2:
            raise TypeError(f"Window size = {self.window_size} is too small for the polynomials order = {self.order}")

    def _check_data(self, data: Any) -> np.ndarray:
        """
        Check validness of derivative selection.
        """
        if not (isinstance(data, np.ndarray) or isinstance(data, list) or isinstance(data, tuple)):
            self._logger.error(f"Provided data should a type of `np.ndarray`, or array-like: `list` or `tuple`, ",
                               f" provided parameter is of type: {type(data)}")
            raise AttributeError(f"Provided data should a type of `np.ndarray`, or array-like: `list` or `tuple`")

        if isinstance(data, list) or isinstance(data, tuple):
            try:
                data = np.asarray(data)
            except Exception as e:
                self._logger.error(f"Provided data should an array-like: `list` or `tuple`,"
                                   " which is convertible to `np.ndarray` type.",
                                   f" Provided parameter is of type: {type(data)} and raised error:\n{e}")
                raise AttributeError(f"Provided data should be convertible to `np.ndarray`!\nIt raised error:\n{e}")
        return data

    def compute_velocity(self, data: Union[np.ndarray, List[Union[float, int]], torch.Tensor], **kwargs) -> Any:
        """
        Computes gaze velocity function.
        """
        order_range = range(self.order + 1)
        half_window = (self.window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[self.derivative.value] * self.rate ** self.derivative.value * factorial(
            self.derivative.value)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = data[0] - np.abs(data[1:half_window + 1][::-1] - data[0])
        lastvals = data[-1] + np.abs(data[-half_window - 1:-1][::-1] - data[-1])
        data = np.concatenate((firstvals, data, lastvals))
        return np.convolve(m[::-1], data, mode='valid')

    def compute_velocity_sample(self, sample: Sample, **kwargs) -> Sample:
        """
        Computes gaze velocity for single Sample.
        """
        sample_data = self._check_data(sample.data)  # initial of size [data_sample_length, n_dims]
        n_dims = sample_data.shape[-1]

        data_ = []
        for dim in range(n_dims):
            data_.append(self.compute_velocity(sample_data[:, dim]))
        data_ = np.vstack(data_)
        data_ = np.einsum('ij->ji', data_)

        # set data for sample
        sample.data = data_
        return sample
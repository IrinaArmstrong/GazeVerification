import torch
import numpy as np
from math import factorial
from typeguard import typechecked
from typing import Union, Any, List, Optional

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.velocity_estimators.velocity_estimator_abstract import VelocityEstimatorAbstract


@typechecked
class SimpleDiffVelocityEstimator(VelocityEstimatorAbstract):
    """
    The gaze sequence of absolute yaw 𝑥 and pitch gaze angles 𝑦 of the left L and right eye R
    recorded with sampling frequency 𝜌 in Hz is transformed into sequences of yaw δ𝑥𝑖 and pitch δ𝑦𝑖.
And gaze velocities in °/s are calculated as follows:

δ𝑥𝑖=𝜌2(𝑥𝑖+1−𝑥𝑖−1)δ𝑦𝑖=𝜌2(𝑦𝑖+1−𝑦𝑖−1)
    """

    def __init__(self,
                 sampling_frequency: float,
                 verbose: bool = True):
        super().__init__(verbose)
        self.sampling_frequency = sampling_frequency
        self._hyperparameters = self.register_hyperparameters()

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
                data = np.asarray(data, dtype=np.float64)
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
        data = np.float64(data)
        velocity = (self.sampling_frequency / 2) * (np.roll(data, -1) -
                                                    np.roll(data, +1))
        # Zero the first and the last one elements
        # for input and output array's sizes consistency
        velocity[[0, -1]] = 0.0
        return velocity

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

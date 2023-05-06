import dataclasses

import torch
import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from typing import Union, Optional, Any, Dict

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.algorithm_abstract import AlgorithmAbstract


@typechecked
class Scaler(AlgorithmAbstract):
    """
    Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between zero and one.
    """

    def __init__(self,
                 min_value: Union[float, int, list, tuple],
                 max_value: Union[float, int, list, tuple],
                 clip: Optional[bool] = False,
                 copy: bool = True):
        """
        :param min_value: lower value(-s) of desired range of transformed data.
                    A different value can be provided for each data channel;
        :type min_value: Union[float, int, list, tuple];
        :param max_value: upper value(-s) of desired range of transformed data.
                    A different value can be provided for each data channel;
        :type max_value: Union[float, int, list, tuple];
        :param clip: Set to True to clip transformed values of held-out data to provided feature range;
        :type clip: bool;
        :param copy: Set to False to perform inplace row normalization and avoid a copy;
        :type copy: bool.
        """
        super().__init__()
        self._copy = copy
        self.min_value = min_value
        self.max_value = max_value
        self._do_clip = clip
        self._do_copy = copy
        self._hyperparameters = self.register_hyperparameters()

    def _scale(self, data: np.ndarray, axis: Optional[int] = 0):
        std = (data - data.min(axis=axis)) / (data.max(axis=axis) - data.min(axis=axis))
        return std * (self.max_value - self.min_value) + self.min_value

    def scale_sample(self, sample: Sample, axis: int) -> Sample:
        """
        Scale data sequences from Samples according to predefined logic.

        :param sample: Sample object containing information about one Sample;
        :type sample: Sample;

        :return: Sample object with scaled data field,
        :rtype: Sample
        """
        sample_data = self._check_data(sample.data)  # initial of size [data_sample_length, n_dims]
        # Axis selection is relevant for horizontal vs. vertical data flipping
        scaled_data = self._scale(data=sample_data,
                                  axis=axis)
        # optionally clipping value
        if self._do_clip:
            scaled_data = np.clip(scaled_data, self.min_value, self.max_value)

        # set data for sample
        if self._copy:
            return dataclasses.replace(sample, data=scaled_data)
        sample.data = scaled_data
        return sample

    def scale_samples(self, samples: Samples, axis: int) -> Samples:
        """
        Scale data sequences from Samples according to predefined logic.

        :param samples: un-scaled data of shape: [n_samples, n_dims] or [n_dims, n_samples].
        :type samples: array-like,

        :param axis: {0, 1}, default=1
                    Define axis used to scale the data along.
                    If 1, independently scale each sample, otherwise (if 0) scale each feature.
        :type axis: int,

        :return: dataset as Samples object with scaled data fields in each Sample,
        :rtype: Samples.
        """
        scaled_samples = []
        for sample in tqdm(samples, total=len(samples), desc="Scaling dataset..."):
            try:
                scaled_samples.append(
                    self.scale_sample(sample, axis=axis))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset scaling: {e}"
                                   f"\nSkipping sample: {sample.guid}.")
                # keep sample even if it's data is not correct
                if self.keep_erroneous_samples:
                    scaled_samples.append(sample)

        return Samples(scaled_samples)

    def run(
            self,
            data: Samples,
            axis: int = 0,
            **kwargs
    ) -> Samples:
        """
        Performs a min-max scaling on the data samples.
        The transformation is given by:

            X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X_scaled = X_std * (max - min) + min

        where min, max = feature_range.

        :param data: unscaled data of shape: [n_samples, n_dims] or [n_dims, n_samples].
        :type data: array-like,

        :param axis: {0, 1}, default=1
                    Define axis used to scale the data along.
                    If 1, independently scale each sample, otherwise (if 0) normalize each feature.
        :type axis: int,

        :return: dataset as Samples object with scaled data fields in each Sample,
        :rtype: Samples.
        """
        return self.scale_samples(samples=data,
                                  axis=axis)

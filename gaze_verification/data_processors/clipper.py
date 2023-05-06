import dataclasses

import torch
import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from typing import Union, Optional, Any, Dict

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.algorithm_abstract import AlgorithmAbstract


@typechecked
class Clipper(AlgorithmAbstract):
    """
    This estimator clips features sequence to a given range.

    Given an interval, values outside the interval are clipped to the interval edges.
    For example, if an interval of [0, 1] is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.
    """

    def __init__(self,
                 lower_bound: Union[float, int],
                 upper_bound: Union[float, int],
                 copy: bool = True):
        """
        :param lower_bound: lower value of desired range of transformed data;
        :type lower_bound: Union[float, int];
        :param upper_bound: upper value of desired range of transformed data;
        :type upper_bound: Union[float, int];
        :param copy: Set to False to perform inplace row normalization and avoid a copy;
        :type copy: bool.
        """
        super().__init__()
        self._copy = copy
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._do_copy = copy
        self._hyperparameters = self.register_hyperparameters()

    def clip_sample(self, sample: Sample) -> Sample:
        """
        Clip data sequences from Samples.

        :param sample: Sample object containing information about one Sample;
        :type sample: Sample;

        :return: Sample object with clipped data field,
        :rtype: Sample
        """
        sample_data = self._check_data(sample.data)  # initial of size [data_sample_length, n_dims]
        # Axis selection is relevant for horizontal vs. vertical data flipping
        clipped_data = np.clip(sample_data, self.lower_bound, self.upper_bound)
        # optionally clipping value
        if self._do_clip:
            clipped_data = np.clip(clipped_data, self.min_value, self.max_value)

        # set data for sample
        if self._copy:
            return dataclasses.replace(sample, data=clipped_data)
        sample.data = clipped_data
        return sample

    def clip_samples(self, samples: Samples) -> Samples:
        """
        Clip data sequences from Samples according to predefined logic.

        :param samples: un-clipped data of shape: [n_samples, n_dims] or [n_dims, n_samples].
        :type samples: array-like,

        :return: dataset as Samples object with scaled data fields in each Sample,
        :rtype: Samples.
        """
        clipped_samples = []
        for sample in tqdm(samples, total=len(samples), desc="Scaling dataset..."):
            try:
                clipped_samples.append(
                    self.clip_sample(sample))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset data clipping: {e}"
                                   f"\nSkipping sample: {sample.guid}.")
                # keep sample even if it's data is not correct
                if self.keep_erroneous_samples:
                    clipped_samples.append(sample)

        return Samples(clipped_samples)

    def run(
            self,
            data: Samples,
            **kwargs
    ) -> Samples:
        """
        Performs a min-max clipping on the data samples.
        Equivalent to but faster than np.minimum(a_max, np.maximum(a, a_min)).

        :param data: raw data of shape: [n_samples, n_dims] or [n_dims, n_samples].
        :type data: array-like,

        :return: dataset as Samples object with clipped data fields in each Sample,
        :rtype: Samples.
        """
        return self.clip_samples(samples=data)
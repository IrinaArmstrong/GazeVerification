import dataclasses

import torch
import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from typing import Union, Optional, Any

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.algorithm_abstract import AlgorithmAbstract


@typechecked
class ZScoreNormalizer(AlgorithmAbstract):
    """
    Z-score normalization refers to the process of normalizing every value in a dataset
    such that the mean of all of the values is 0 and the standard deviation is 1.

    Here is used the following formula to perform a z-score normalization on every value in a data:
        new_value = (old_value – μ) / σ
    where:
        old_value: Original value,
        μ: Mean of data,
        σ: Standard deviation of data.

    In other words, for each sample from the dataset, we subtract the mean and divide by the standard deviation.
    By removing the mean from each sample, we effectively move the samples towards a mean of 0
    (after all, we removed it from all samples).
    In addition, by dividing by the standard deviation, we yield a dataset
    where the values describe by how much of the standard deviation they are offset from the mean.

    Note:
    In some cases it may be useful to use floating-point numbers with more precision then provided in input data,
    otherwise some values can appear to be zero during std computation.
    Main purpose is (optionally) convert input data (from float16) to higher precision types
    (for example to float32) for doing computation and then return initial type for returning value.
    """

    def __init__(self,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 copy: bool = True):
        """
        :param mean: pre-defined mean,
        :type mean: float,
        :param std: pre-defined standard deviation,
        :type std: float,
        :param copy: set to False to perform inplace data normalization and avoid a copy,
        :type copy: bool, default=True
        """
        super().__init__()
        self._copy = copy
        self.mean = mean
        self.std = std

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

    def _normalize(self, data: Union[np.ndarray, torch.Tensor],
                   axis: int, inner_dtype: type, output_dtype: type,
                   ) -> np.ndarray:
        """
        Performs a z-score normalization on the data as array or tensor.
        :param data: un-normalized data,
        :type data: array or tensor,
        :return: normalized data,
        :rtype: array.
        """
        if self.mean is None:
            mu = np.mean(data, axis=axis, dtype=inner_dtype)
        else:
            mu = self.mean
        if self.std is None:
            std = np.std(data, axis=axis, dtype=inner_dtype)
        else:
            std = self.std
        return ((data - mu) / std).astype(output_dtype)

    def normalize_sample(self, sample: Sample, axis: int,
                         inner_dtype: type, output_dtype: type) -> Sample:
        """
        Filter data sequences from Samples according to predefined logic.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: Sample object with normalized data field,
        :rtype: Sample
        """
        sample_data = self._check_data(sample.data)  # initial of size [data_sample_length, n_dims]
        # Axis selection is relevant for horizontal vs. vertical data flipping
        normalized_data = self._normalize(data=sample_data,
                                          axis=axis,
                                          inner_dtype=inner_dtype,
                                          output_dtype=output_dtype)
        # set data for sample
        if self._copy:
            return dataclasses.replace(sample, data=normalized_data)
        sample.data = normalized_data
        return sample

    def normalize_samples(self, samples: Samples, axis: int,
                          inner_dtype: type, output_dtype: type) -> Samples:
        """
        Filter data sequences from Samples according to predefined logic.

        :param samples: un-normalize data of shape: [n_samples, n_dims] or [n_dims, n_samples].
        :type samples: array-like,

        :param axis: {0, 1}, default=1
                    Define axis used to normalize the data along.
                    If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
        :type axis: int,

        :param inner_dtype: inner data type, usually a floating-point precision higher then provided in input data,
        :type inner_dtype: type,

        :param output_dtype: output data type,
        :type output_dtype: type,

        :return: dataset as Samples object with normalized data fields in each Sample,
        :rtype: Samples.
        """
        normalized_samples = []
        for sample in tqdm(samples, total=len(samples), desc="Normalizing dataset..."):
            try:
                normalized_samples.append(
                    self.normalize_sample(sample, axis=axis, inner_dtype=inner_dtype, output_dtype=output_dtype))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset normalization: {e}"
                                   f"\nSkipping sample: {sample.guid}.")
                # keep sample even if it's data is not filtered
                if self.keep_erroneous_samples:
                    normalized_samples.append(sample)

        return Samples(normalized_samples)

    def run(
            self,
            data: Samples,
            axis: int = 0,
            inner_dtype: Optional[type] = np.float32,
            output_dtype: Optional[type] = np.float16,
            **kwargs
    ) -> Samples:
        """
        Performs a z-score normalization on the data samples.
         Here is used the following formula to perform a z-score normalization on every value in a data:
            new_value = (old_value – μ) / σ
        where:
            old_value: Original value,
            μ: Mean of data,
            σ: Standard deviation of data.

        :param data: un-normalize data of shape: [n_samples, n_dims] or [n_dims, n_samples].
        :type data: array-like,

        :param axis: {0, 1}, default=1
                    Define axis used to normalize the data along.
                    If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
        :type axis: int,

        :param inner_dtype: inner data type, usually a floating-point precision higher then provided in input data,
        :type inner_dtype: type,

        :param output_dtype: output data type,
        :type output_dtype: type,

        :return: dataset as Samples object with normalized data fields in each Sample,
        :rtype: Samples.
        """
        return self.normalize_samples(samples=data,
                                      axis=axis,
                                      inner_dtype=inner_dtype,
                                      output_dtype=output_dtype)

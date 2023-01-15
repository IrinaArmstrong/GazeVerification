import torch
import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing import Union, Any, List

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.algorithm_abstract import AlgorithmAbstract


@typechecked
class VelocityEstimatorAbstract(AlgorithmAbstract, ABC):
    """
    Abstract class for all gaze velocity estimators.

    Functionality:
    Estimate instantaneous gaze velocity using a provided processing function.
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self._verbose = verbose

    @abstractmethod
    def compute_velocity(self, data: Union[np.ndarray, List[Union[float, int]], torch.Tensor], **kwargs) -> Any:
        """
        Computes gaze velocity function.
        """
        return NotImplementedError

    @abstractmethod
    def compute_velocity_sample(self, sample: Sample, **kwargs) -> Sample:
        """
        Computes gaze velocity for single Sample.
        """
        return NotImplementedError

    def compute_velocity_samples(self, samples: Samples, **kwargs) -> Samples:
        """
        Computes gaze velocity for all Samples.
        """
        samples_ = []
        for sample in tqdm(samples, total=len(samples), desc="Velocity computation for a dataset..."):
            try:
                samples_.append(
                    self.compute_velocity_sample(sample, **kwargs))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset velocity computation: {e}"
                                   f"\nSkipping sample: {sample.guid}.")
                # keep sample even if it's data is not filtered
                if self.keep_erroneous_samples:
                    samples_.append(sample)

        return Samples(samples_)

    def run(self, data: Samples, **kwargs) -> Samples:
        """
        Process Samples to create new ones with velocities data.

        :param data: Samples containing N Samples
        :type data: Samples

        :return: Samples object containing N Samples
        :rtype: Samples
        """
        dataset = self.compute_velocity_samples(data)
        return dataset

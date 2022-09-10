from abc import ABC, abstractmethod

from gaze_verification.logging_handler import get_logger
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_objects.sample import Sample, Samples


class FilterAbstract(AlgorithmAbstract, ABC):
    """
    Raw eye gaze positions may contain noise. The presence of noise makes it difficult to estimate
    the velocity and acceleration parameters using differentiation operation.
    Thus, filtering is a necessary operation for the stable algorithms performance.

    Abstract class for all data filters.
    Functionality:
    Define the interface for most part of methods for data Samples segmentation process.

    :param verbose: Turn on/off logging
    :type verbose: bool
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self._verbose = verbose
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="DEBUG" if self._verbose else "INFO"
        )

    def run(self, data: Samples, **kwargs) -> Samples:
        """
        Process Instances to create new segmented ones.

        :param data: Samples containing N filtered Samples
        :type data: Samples

        :return: Samples object containing N filtered Samples
        :rtype: Samples
        """
        dataset = self.filter_dataset(data)
        return dataset

    @abstractmethod
    def filter_dataset(self, samples: Samples) -> Samples:
        """
        Create a new dataset containing filtered Samples.

        :param samples: DataClass containing N filtered Samples
        :type samples: Instances

        :return: Samples object containing N filtered Samples
        :rtype: Samples
        """
        raise NotImplementedError

    @abstractmethod
    def filter_sample(self, sample: Sample) -> Sample:
        """
        Filter data sequences from Samples according to predefined logic.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: Sample object with filtered data field,
        :rtype: Sample
        """
        raise NotImplementedError

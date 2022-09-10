from abc import ABC, abstractmethod
from typing import List

from gaze_verification.logging_handler import get_logger
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_objects.sample import Sample, Samples


class SegmentorAbstract(AlgorithmAbstract, ABC):
    """
    Abstract class for all data segmentors.

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

        :param data: Samples containing N formatted Samples
        :type data: Samples

        :return: Samples object containing N formatted Samples
        :rtype: Samples
        """
        dataset = self.build_segmented_dataset(data)
        return dataset

    @abstractmethod
    def build_segmented_dataset(self, samples: Samples) -> Samples:
        """
        Create a new dataset containing segmented and formatted Samples.

        :param samples: DataClass containing N formatted Samples
        :type samples: Instances

        :return: Samples object containing N formatted Samples
        :rtype: Samples
        """
        raise NotImplementedError

    @abstractmethod
    def build_segments(self, sample: Sample) -> List[Sample]:
        """
        Segment data sequences from Samples on M parts according to predefined logic.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: List with N formatted Samples
        :rtype: List[Sample]
        """
        raise NotImplementedError

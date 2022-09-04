import torch
import numpy as np
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects import Samples


@typechecked
class AlgorithmAbstract(ABC):
    """
    An abstract class for all algorithms.
    Algorithm is something that can be used as part of the pipeline.
    """

    def __init__(self):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

    @abstractmethod
    def run(
            self,
            data: Union[np.ndarray, List[float], Samples, torch.Tensor, str],
            **kwargs
    ) -> Optional[Union[Samples, str, Any]]:
        """
        Any custom data transform.

        :param data:
            Samples - a list of Samples,
            np.ndarray or List[float] or torch.Tensor - an array-like of raw data,
            str - a path to the data file or folder,

        :type data: Union[np.ndarray, List[float], Samples, torch.Tensor, str]

        :return: result of the algorithm's work.
        :rtype: Optional[Samples]
        """
        raise NotImplementedError

    @staticmethod
    def set_output_dir(output_dir: str = "./"):
        """
        Set an output directory.
        :param output_dir: where to output (logs or visualizations), defaults to "./"
        :type output_dir: str
        :return:
        """
        pass

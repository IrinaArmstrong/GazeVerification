import torch
import enum
import numpy as np
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, Dict

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects import Samples


@typechecked
class AlgorithmAbstract(ABC):
    """
    An abstract class for all algorithms.
    Algorithm is something that can be used as part of the pipeline.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self._hyperparameters = self.register_hyperparameters(**kwargs)

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

    def register_hyperparameters(self, names: List[str] = None) -> List[str]:
        """
        Register hyperparameters of an algorithm.
        :param names: optional input hyperparameter's names;
        :type names: a list of str;
        """
        hyperparameters_ = []
        for key, val in self.__dict__.items():
            if not key.startswith("_"):
                hyperparameters_.append(key)
        if names is not None:
            hyperparameters_ = list(set(hyperparameters_) & set(names))
        return hyperparameters_

    def get_hyperparameters(self, as_dict: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Returns hyperparameters of an algorithm and their values.
        :param as_dict: return them as a dictionary,
                        with a key - parameter name, value - it's value;
                        Otherwise a concatenated string of type:
                            `parameter_name_1=value__parameter_name_2=value__...`
                        would be returned.
        :type as_dict: bool;
        """
        hyperparams_dict = {}
        for hp_name in self._hyperparameters:
            if hasattr(self, hp_name):
                hp_value = getattr(self, hp_name)
                if isinstance(hp_value, enum.Enum):
                    hyperparams_dict[hp_name] = hp_value.value
                else:
                    hyperparams_dict[hp_name] = hp_value
        if not len(hyperparams_dict):
            self._logger.error(f"No hyperparameters found!")
        if as_dict:
            return hyperparams_dict
        # as a single string:
        # `parameter_name_1=value__parameter_name_2=value__...`
        hyperparams_str = ""
        for hp_name, hp_value in hyperparams_dict.items():
            hyperparams_str += f"{hp_name}={hp_value}__"
        return hyperparams_str.strip("_")

    def save_hyperparameters(self, filename: str, extension: str = "json",
                             as_dict: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Saves hyperparameters of an algorithm and their values to the file.
        :param filename: a name of the file for hyperparameters saving;
        :type filename: str;
        :param extension: an extension of the file;
        :type extension: str, default - json;
        :param as_dict: return them as a dictionary,
                        with a key - parameter name, value - it's value;
                        Otherwise a concatenated string of type:
                            `parameter_name_1=value__parameter_name_2=value__...`
                        would be returned.
        :type as_dict: bool;
        """
        raise NotImplementedError

import os
import enum
import torch

from abc import abstractmethod, ABC
from typeguard import typechecked
from typing import Dict, Any, Type, Union

from gaze_verification.logging_handler import get_logger


@typechecked
class EmbedderAbstract(torch.nn.Module, ABC):
    """
    Abstract class for making embeddings out of tokenized texts.

    The method create_inputs vectorizes the data for one sample,
    and collate_fn forms tensors inside a batch.
    """
    def __init__(self, config: "EmbedderConfigAbstract", *args, **kwargs):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.name: str
        self.config = config
        self.model: torch.nn.Module

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model.forward(*args, **kwargs)

    @property
    def name(self):
        if self.name is None:
            return self.__class__.__name__
        else:
            return self.name

    @abstractmethod
    def get_hidden_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings_size(self) -> int:
        raise NotImplementedError


class EmbedderConfigAbstract(ABC):
    """
    This class is the ancestor of all the classes that defines configurations for embedders.

    :param get_target_class: returns the class that is used this config class
    :param model_type: type of the model that uses this config class
    """
    model_type: str

    def __init__(self):
        self.check_model_type()

    @staticmethod
    def get_target_class() -> Type[EmbedderAbstract]:
        """
        :return: the class that is used this config class
        """
        raise NotImplementedError()

    @classmethod
    def check_model_type(cls):
        if cls.model_type is None:
            raise ValueError("Attribute 'model_type must be set, but got None")

    def get_parameters(self, as_dict: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Returns parameters from the config and their values.
        :param as_dict: return them as a dictionary,
                        with a key - parameter name, value - it's value;
                        Otherwise a concatenated string of type:
                            `parameter_name_1=value__parameter_name_2=value__...`
                        would be returned.
        :type as_dict: bool;
        """
        params_dict = {}
        for param_name, param_value in self.__dict__.items():
            if param_name.startswith("_"):
                continue
            if isinstance(param_value, enum.Enum):
                params_dict[param_name] = param_value.value
            else:
                params_dict[param_name] = param_value
        if as_dict:
            return params_dict
        # as a single string:
        # `parameter_name_1=value__parameter_name_2=value__...`
        params_str = ""
        for param_name, param_value in params_dict.items():
            params_str += f"{param_name}={param_value}__"
        return params_str.strip("_")

    def save_configuration(self, filename: str, extension: str = "json",
                           as_dict: bool = True, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Saves config parameters of an algorithm and their values to the file.
        :param filename: a name of the file for parameters saving;
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
        # Check parent folder exists
        if not os.path.isdir(os.path.dirname(filename)):
            raise NotADirectoryError(f"Directory for file saving do not exists"
                                     f"or is not a valid directory: {os.path.dirname(filename)}")
        # Select parameters
        content = self.get_parameters(as_dict)

        # Save
        if extension == "json":
            with open(filename, "w", encoding="utf-8") as f:
                # **kwargs defaults: {'skipkeys': False, 'ensure_ascii': True,
                # 'allow_nan': True, 'indent': None, 'separators': None, 'sort_keys': False}
                f.write(content, **kwargs)
        elif extension == "txt":
            with open(filename, "wb") as f:
                f.write(content, **kwargs)
        else:
            raise ValueError(f"Unknown extension {extension}!")

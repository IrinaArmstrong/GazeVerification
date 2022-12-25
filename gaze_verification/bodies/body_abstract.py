import torch

from abc import abstractmethod, ABC
from torch import nn
from typeguard import typechecked
from typing import Optional, Type, Dict, Any


@typechecked
class BodyAbstract(nn.Module, ABC):
    """
    Abstract class for postprocessing embeddings from embedder.
    """

    def __init__(
            self, config
    ):
        """
        :param config: a configuration parameters for creating Body instance;
        :type config: BodyConfigAbstract.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, embedder_output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class BodyConfigAbstract(ABC):
    """
    This class is the ancestor of all the classes that defines configurations for bodies.
    """

    def __init__(
            self,
            input_size: Optional[int] = None,
            hidden_size: Optional[int] = None,
            **kwargs
    ):
        """
        :param input_size: a dimension of the input vectors;
        :type input_size: int;
        :param hidden_size: a dimension of the output vectors. By default is set to None.
                            If None, it sets up to input_size;
        :type hidden_size: Optional[int].
        """
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size

    @staticmethod
    @abstractmethod
    def get_target_class() -> Type[BodyAbstract]:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )

import torch

from torch import nn
from typing import Optional, Type, Dict, Any

from gaze_verification.bodies.body_abstract import BodyAbstract, BodyConfigAbstract


class EmptyBody(BodyAbstract):
    """
    Base and the simplest class for all bodies. It consists of at most one linear layer.
    Input and output sizes are configured as follows:
        [batch_size x seq_len_input x input_size] -> [batch_size x seq_len_input x hidden_size]
    """
    def __init__(
            self,
            config,
            **kwargs
    ):
        super().__init__(config, **kwargs)

        if self.config.hidden_size != self.config.input_size:
            self.linear = nn.Linear(self.input_size, self.hidden_size)
        else:
            self.linear = nn.Identity()

    def forward(self, embedder_output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.linear(embedder_output)


class EmptyBodyConfig(BodyConfigAbstract):
    """
    Config for EmptyBody.
    """

    def __init__(
            self,
            input_size: Optional[int] = None,
            hidden_size: Optional[int] = None,
            **kwargs
    ):
        super().__init__(input_size, hidden_size)

    @staticmethod
    def get_target_class() -> Type[BodyAbstract]:
        return EmptyBody

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

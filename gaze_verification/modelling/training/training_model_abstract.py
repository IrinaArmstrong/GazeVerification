from abc import abstractmethod
from typeguard import typechecked
from typing import Optional, List, Any, Union, Dict

import torch
from torch.optim import Optimizer
from pytorch_lightning import LightningModule


@typechecked
class TrainingModelAbstract(LightningModule):
    """
    Abstract model for all models, which can run in fit-/eval-mode.
    Current class describes interface.
    """
    def __init__(self):
        super().__init__()
        self.predictions = []

        # optimizers params
        self._learning_rate = None
        self._optimizer = None
        self._optimizer_kwargs = None

        # scheduler params
        self._scheduler_name = None
        self._total_steps = None
        self._scheduler_kwargs = None

    def configure_optimizers(self) -> Dict[str, Any]:
        pass

    def set_optimizer_params(
            self,
            optimizer_params: Dict[str, Any],
            optimizer: Optional[Optimizer] = None,
    ) -> None:
        pass

    def set_scheduler_params(
            self,
            total_steps: int,
            scheduler_name: str = "linear",
            **scheduler_kwargs
    ) -> None:
        pass

    def test_step(
            self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def training_step(
            self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Describe training step, as it is required in pytorch_lightning
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(
            self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Describe validation step, as it is required in pytorch_lightning
        """
        raise NotImplementedError



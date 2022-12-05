import torch
from abc import abstractmethod, ABC
from typeguard import typechecked
from typing import Dict, List

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects import (Sample, Samples, Label)
from gaze_verification.target_configurators.target_configurator import TargetConfigurator


@typechecked
class PredictorAbstract(torch.nn.Module, ABC):
    """
    Abstract predictor class that contains task-specific methods.
    Prepares input labels and masks for a model, gets a loss, converts tensors to predictions.
    """

    POOLERS = ("first", "max", "relu-max", "sum", "avg")

    def __init__(self, targets_config_path: str, hidden_size: int,
                 confidence_level: float = 0.5,
                 class_weights: Dict[int, float] = None,
                 p_dropout: float = 0.6):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.p_dropout = p_dropout
        self.confidence_level = confidence_level
        self.hidden_size = hidden_size
        self.class_weights = class_weights

        self.configurator = TargetConfigurator(targets_config_path)
        self.head = None  # must be defined for each child

    @abstractmethod
    def create_targets(self, samples: Sample, is_predict: bool = False) -> Samples:
        """
        Take the fields responsible for target's labels from the sample and converts them into indices.
        :param samples: data samples;
        :type samples: Samples;
        :param is_predict: whether model is running in predict mode;
        :type is_predict: bool, defaults to False;
        :return: data samples with targets labels converted to indexes;
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError

    @abstractmethod
    def convert_batch_outputs_to_predictions(
            self,
            predictions: torch.Tensor,
            probas: torch.Tensor,
            *args, **kwargs
    ) -> List[Label]:
        """
        Convert batch tensors into predictions.
        :param predictions: label predictions;
        :type predictions: torch.Tensor;
        :param probas: label probabilities;
        :type probas: torch.Tensor;
        :return: predictions for the batch;
        :rtype: List[Label].
        """
        raise NotImplementedError

    @abstractmethod
    def set_predicted_label(
            self,
            sample: Sample,
            predicted_markup: Label,
            inplace: bool = True
    ) -> Sample:
        raise NotImplementedError

    def save_predictions_to_samples(
            self,
            samples: Samples,
            predictions: List[Label],
            inplace: bool = True
    ) -> Samples:
        """
        Save model predictions into instances.
        :param samples: samples to update;
        :type samples: Samples;
        :param predictions: model predictions.
        :type predictions: List[MarkupAbstract]
        :return: updated Samples;
        :rtype: Samples.
        """
        updated_samples = []
        for sample, prediction in zip(samples, predictions):
            sample = self.set_predicted_label(sample, prediction, inplace)
            updated_samples.append(sample)
        return Samples(updated_samples)

    @abstractmethod
    def convert_sample_outputs_to_predictions(self, *args, **kwargs) -> Label:
        """
        Convert sample outputs into predictions.
        :return: predicted label for the sample.
        :rtype: Label object.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
            self,
            label_logits,
            labels
    ) -> torch.Tensor:
        """
        Compute loss given the output tensors.
        :param label_logits: label logits;
        :type label_logits: Optional[torch.Tensor], defaults to None;
        :param labels: labels;
        :type labels: Optional[torch.Tensor], defaults to None;
        :return: calculated loss value;
        :rtype: torch.Tensor.
        """
        raise NotImplementedError

    def _create_labels_ids(
            self, n_samples: int, max_len_target: int, data: list
    ) -> torch.Tensor:
        labels_batch = torch.zeros(
                (n_samples, max_len_target), dtype=torch.int64
            )
        return labels_batch



import torch
from abc import ABC, abstractmethod
from torch import nn
from typing import Tuple, Union, Optional, Any


class HeadAbstract(nn.Module, ABC):
    """
    Abstract class for all heads.
    """

    def __init__(self,
                 hidden_size: Optional[int]):
        super().__init__()
        self.hidden_size = hidden_size

    @abstractmethod
    def forward(
            self, body_output: torch.Tensor, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def score(
            self, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict_from_logits(
            self,
            label_logits: torch.Tensor,
            *args, **kwargs
    ) -> Union[torch.Tensor, Any]:
        """
        Retrieves prediction labels from logits.
        :param label_logits: logits vector/tensor from model outputs;
        :type label_logits: torch.Tensor;
        :return: predictions;
        :rtype: torch.Tensor or any other type;
        """
        raise NotImplementedError


class LinearClassificationHead(HeadAbstract):
    """
    Simple linear single layer head for prediction fixed number of classes.
    """

    def __init__(
            self,
            input_size: int,
            num_classes: int,
            p_dropout: float = 0.2,
            confidence_level: float = 0.5,
            class_weights: Optional[torch.Tensor] = None,
            label_smoothing: Optional[float] = None,
    ):
        super().__init__(input_size)
        self.num_classes = num_classes
        self.binary = self.num_classes == 2
        self.confidence_level = confidence_level

        self.dropout = nn.Dropout(p_dropout)
        self.label_classifier = nn.Linear(self.input_size, self.num_classes)

        # Два лосса используются потому, что поведение может быть разным,
        # когда маска есть и когда ее нет.

        if self.binary:
            # for binary classification better to use BCE
            loss_fn = nn.BCEWithLogitsLoss
            self.sigmoid = nn.Sigmoid()
        else:
            # for multiclass
            loss_fn = nn.CrossEntropyLoss

        kwargs = {}
        kwargs["label_smoothing"] = label_smoothing or 0.0

        self.loss_mean = loss_fn(reduction="mean", weight=class_weights, **kwargs)

    def forward(self, body_output: torch.Tensor, **kwargs) -> torch.Tensor:
        label_logits = self.label_classifier(self.dropout(body_output))
        return label_logits

    def score(
            self,
            label_logits: torch.Tensor,
            labels: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        """
        Считает лосс.
        :param label_logits: логиты (batch_size, seq_len, n_labels)
        :param labels: правильные метки (batch_size, seq_len, n_labels) or (batch_size, seq_len)
        :param label_mask: маска (batch_size, seq_len) or None
        :return: лосс
        """

        label_logits = label_logits.view(-1, self.num_labels)
        labels = labels.view(-1)  # (batch_size * seq_len)
        loss = self.loss_mean(label_logits, labels)
        return loss

    def predict_from_logits(
            self,
            label_logits: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate predictions - i.e. most probable class based on probabilities.
        Here probabilities are softmaxed logits.
        :param label_logits: logits vector, of size: [bs, *] - ?;
        :type label_logits: torch.Tensor;
        :return:
        :rtype:
        """
        probas = torch.softmax(label_logits, dim=-1)
        # tuple of (max scores, max score's indexes == classes)
        scores, preds = label_logits.max(dim=-1)
        return scores, probas, preds
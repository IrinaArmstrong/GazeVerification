import torch
import numpy as np
from typeguard import typechecked
from typing import (Dict, Any, List, Tuple,
                    Optional, Type, Union)

from gaze_verification.data_objects import (Sample, Samples, Label, Target,
                                            ClassificationLabel, ClassificationTarget)
from gaze_verification.predictors.predictor_abstract import PredictorAbstract
from gaze_verification.modelling.heads import LinearClassificationHead


@typechecked
class ClassificationPredictor(PredictorAbstract):
    """
    Classification Predictor class that contains task-specific methods.
    :param config: config for the targeter.
    :type config: ClfTargeterAbstractConfig
    """

    def __init__(self, targets_config_path: str,
                 hidden_size: int,
                 confidence_level: float = 0.5,
                 class_weights: Dict[int, float] = None,
                 p_dropout: float = 0.6,
                 label_smoothing: float = 0.0):
        super().__init__(targets_config_path)
        self.p_dropout = p_dropout
        self.confidence_level = confidence_level
        self.hidden_size = hidden_size
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.idx2target = np.array(self.configurator.idx2target)
        self.head = LinearClassificationHead(
            input_size=self.hidden_size,
            num_classes=len(self.idx2target),
            p_dropout=self.p_dropout,
            confidence_level=self.confidence_level5,
            class_weights=self.class_weights,
            label_smoothing=self.label_smoothing
        )

    @property
    def label_class(self) -> Type[Label]:
        return ClassificationLabel

    @property
    def target_class(self) -> Type[Target]:
        return ClassificationTarget

    def compute_loss(
            self,
            label_logits,
            labels,
            **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute loss given the output tensors with head.
        :param label_logits: label logits;
        :type label_logits: Optional[torch.Tensor], defaults to None;
        :param labels: labels;
        :type labels: Optional[torch.Tensor], defaults to None;
        :return: calculated loss value and some additional parameters (optionally);
        :rtype: Union[Dict[str, torch.Tensor], torch.Tensor].
        """
        loss = self.head.score(label_logits, labels)
        return loss

    def get_logits(
            self,
            body_output: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        logits = self.head.forward(body_output, **kwargs)
        return logits

    def convert_batch_outputs_to_predictions(
            self,
            label_preds_batch: torch.Tensor,
            label_probas_batch: torch.Tensor,
            **kwargs
    ) -> List[ClassificationLabel]:
        """
        Convert batch tensors into predictions.

        :param label_preds_batch: label predictions.
        :type label_preds_batch: torch.Tensor

        :param label_probas_batch: label probabilities.
        :type label_probas_batch: torch.Tensor

        :return: predictions for the batch.
        :rtype: List[ClassificationMarkup]
        """

        batch_size = label_preds_batch.shape[0]

        label_preds_batch = label_preds_batch.detach().cpu().numpy()
        label_probas_batch = label_probas_batch.detach().cpu().numpy()

        predictions_batch = []
        for i in range(batch_size):
            predictions_sample = self.convert_sample_outputs_to_predictions(
                label_preds_sample=label_preds_batch[i],
                label_probas_sample=label_probas_batch[i]
            )
            predictions_batch.append(predictions_sample)
        return predictions_batch

    def convert_sample_outputs_to_predictions(
            self,
            label_preds_sample: Union[np.ndarray, np.int64],
            label_probas_sample: np.ndarray
    ) -> ClassificationLabel:
        """
        Convert sample outputs into predictions.
        :param label_preds_sample: sample label predictions.
        :type label_preds_sample: Union[np.ndarray, np.int64]
        :param label_probas_sample: predicted probabilities for sample labels.
        :type label_probas_sample: np.ndarray
        :return: predicted markup for the sample.
        :rtype: ClassificationMarkup
        """
        # TODO: write func!
        pass
        # a shorthand for np.asarray(condition).nonzero().
        # token_idx_iter, = np.where(label_preds_sample)
        # token_label_list = self.idx2target[token_idx_iter].tolist()
        #
        # probas_dict = {}
        # for class_type, proba in zip(self.idx2target, label_probas_sample):
        #     probas_dict[class_type] = float(proba)
        #
        # classes_seq = []
        # for label in token_label_list:
        #     entity_attrs = None
        #     entity_attrs_dict = self.configurator.get_attributes(label)
        #     n_attrs = len(entity_attrs_dict)
        #     if self.configurator.is_onehot(label):
        #         label_idx = self.configurator.entity2idx_ohe[label]
        #         # tensor n_attrs -> n_attrs -> 1
        #         attr_logits = attr_ohe_logits_sample[label_idx, :n_attrs]
        #         argmax_idx = attr_logits.argmax()
        #         for attr in entity_attrs_dict:
        #             if argmax_idx == entity_attrs_dict[attr]["idx"]:
        #                 entity_attrs = [attr]
        #                 continue
        #     elif self.configurator.is_multihot(label):
        #         label_idx = self.configurator.entity2idx_mhe[label]
        #         # tensor n_attrs -> n_attrs
        #         attr_logits = attr_mhe_logits_sample[label_idx, :n_attrs]
        #         # TODO: configure threshold, return probabilities of attrs
        #         attr_type_ids, = np.where(attr_logits >= 0)
        #         attr_type_ids = set(attr_type_ids.tolist())
        #         entity_attrs = []
        #         for attr in entity_attrs_dict:
        #             if entity_attrs_dict[attr]["idx"] in attr_type_ids:
        #                 entity_attrs.append(attr)
        #
        #     entity = self.entity_class(type=label, attributes=entity_attrs)
        #     classes_seq.append(entity)
        #
        # return self.markup_class(classes_seq, probabilities=probas_dict)

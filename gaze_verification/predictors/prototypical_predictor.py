import torch
import numpy as np
from copy import deepcopy
from typeguard import typechecked
from typing import (Dict, Any, List, Tuple,
                    Optional, Type, Union)

from gaze_verification.data_objects import (Sample, Samples, Label, Target,
                                            PrototypicalLabel, ClassificationTarget)
from gaze_verification.predictors.predictor_abstract import PredictorAbstract
from gaze_verification.modelling.heads import PrototypicalHead


@typechecked
class PrototypicalPredictor(PredictorAbstract):
    """
    Prototypical Predictor class that contains task-specific methods.
    :param config: config for the targeter.
    :type config: PrototypicalTargeterAbstractConfig
    """

    def __init__(self, targets_config_path: str,
                 hidden_size: int,
                 embedding_size: int,
                 n_support: int,
                 confidence_level: float = 0.5,
                 p_dropout: float = 0.6,
                 do_compute_loss: Optional[bool] = False,
                 is_predict: bool = False):
        super().__init__(targets_config_path)

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_support = n_support

        self.p_dropout = p_dropout
        self.confidence_level = confidence_level  # todo: implement query classification with threshold

        self.do_compute_loss = do_compute_loss
        self.is_predict = is_predict
        self.head = PrototypicalHead(
            input_size=self.hidden_size,
            output_size=self.embedding_size,
            n_support=n_support,
            p_dropout=self.p_dropout,
            confidence_level=self.confidence_level,
            do_compute_loss=do_compute_loss,
            is_predict=is_predict
        )

    @property
    def label_class(self) -> Type[Label]:
        return PrototypicalLabel

    @property
    def target_class(self) -> Type[Target]:
        return ClassificationTarget

    def compute_loss(
            self,
            label_logits: torch.Tensor,
            labels: torch.Tensor,
            **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute loss given the output tensors with head.
        :param label_logits: label logits;
        :type label_logits: torch.Tensor;
        :param labels: labels;
        :type labels: torch.Tensor;
        :return: calculated loss value;
        :rtype: Union[Dict[str, torch.Tensor], torch.Tensor].
        """
        outputs = self.head.score(label_logits, labels)
        return outputs.get('loss')

    def set_predicted_label(
            self,
            sample: Sample,
            predicted_label_id: Any,
            predicted_probas: Optional[Any] = None,
            distances: Optional[Any] = None,
            inplace: Optional[bool] = True
    ) -> Sample:
        predicted_label = self.convert_sample_outputs_to_predictions(predicted_label_id=predicted_label_id,
                                                                     predicted_probas=predicted_probas,
                                                                     distances=distances)
        if inplace:
            sample.predicted_label = predicted_label
            return sample
        else:
            copied_sample = deepcopy(sample)
            copied_sample.predicted_label = predicted_label
            return copied_sample

    def convert_batch_outputs_to_predictions(
            self,
            batch_predicted_label_ids: torch.Tensor,
            batch_predicted_probas: torch.Tensor,
            batch_distances: Optional[torch.Tensor] = None,
            **kwargs
    ) -> List[PrototypicalLabel]:
        """
        Convert batch predicted labels and attributes into predictions.

        :param batch_predicted_label_ids: predicted labels ids for batch samples;
        :type batch_predicted_label_ids: torch.Tensor;
        :param batch_predicted_probas: predicted labels probabilities for batch samples;
        :type batch_predicted_probas: torch.Tensor;
        :param batch_distances: computed distances matrix [n_queries, n_prototypes] for batch samples;
        :type batch_distances:  torch.Tensor;
        :return: predictions for the batch;
        :rtype: List[Label].
        """

        batch_size = batch_predicted_label_ids.shape[0]

        batch_predicted_label_ids = batch_predicted_label_ids.detach().cpu().numpy()
        batch_predicted_probas = batch_predicted_probas.detach().cpu().numpy()
        if batch_distances is not None:
            batch_distances = batch_distances.detach().cpu().numpy()

        predictions_batch = []
        for i in range(batch_size):
            label_attrs = {
                "predicted_label_id": batch_predicted_label_ids[i],
                "predicted_probas": batch_predicted_probas[i]
            }
            if batch_distances is not None:
                label_attrs['distances'] = batch_distances[i]
            sample_prediction = self.convert_sample_outputs_to_predictions(
                **label_attrs
            )
            predictions_batch.append(sample_prediction)
        return predictions_batch

    def convert_sample_outputs_to_predictions(self, predicted_label_id: Any,
                                              predicted_probas: Optional[Any] = None,
                                              distances: Optional[Any] = None) -> Label:
        """
        Convert sample outputs into predictions.
        :return: predicted label for the sample.
        :rtype: Label object.
        """
        target_name = self.configurator.get_idx2target(predicted_label_id)
        predicted_target = self.target_class(name=target_name,
                                             id=predicted_label_id,
                                             attributes=self.configurator.targets.get(target_name).get("attributes"))
        if not (isinstance(predicted_probas, tuple) or isinstance(predicted_probas, list)):
            predicted_probas = [predicted_probas]
        predicted_label = self.label_class(
            labels=[predicted_target],
            probabilities=predicted_probas,
            distances=distances
        )
        return predicted_label

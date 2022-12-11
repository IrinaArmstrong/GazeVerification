from typing import List
from typeguard import typechecked
from gaze_verification.modelling.training.training_model_abstract import TrainingModelAbstract
from gaze_verification.modelling.inference.inference_model_abstract import InferenceModelAbstract

from auto_ner.bodies.body_abstract import BodyConfigAbstract
from auto_ner.core.instance import Instances
from auto_ner.core.markups import MarkupAbstract
from auto_ner.embedders.embedder_abstract import EmbedderConfigAbstract
from auto_ner.model.training_model_abstract import TrainingModelAbstract
from auto_ner.targeters.targeter_abstract import TargeterAbstractConfig


@typechecked
class Model(InferenceModelAbstract, TrainingModelAbstract):
    """
    Common class for most part of models. It is inherited from LightningModule
    from pytorch_lightning (via TrainingModelAbstract). Last one is used as a common
    and known Trainer with many features and readable documentation.
    Moreover current class includes functionality from InferenceModel
    class - inference-ready pipeline.
    Model class describes training_step, validation_step and
    save_valid_predictions_to_instances, which allows user to fit/eval model
    and predicting with model.

    To use this functionality your model should consist of three main parts:

        - embedder - prepares input sequences and forward it through embedder
        - body - transforms the embedder outputs by contextualizing them.
        - targeter - makes a task-specific prediction

    Examples of embedders: word2vec, ELMo, BERT (embeddings only ot the whole model),
    nn.Embedding.

    Examples of bodies: nn.Identity, MLP, transformers, LSTM, attention.

    Examples of targeters: MLP for tagging or classification, CRF for tagging.
    """

    def __init__(
            self,
            embedder: EmbedderConfigAbstract,
            body: BodyConfigAbstract,
            targeter: TargeterAbstractConfig
    ):
        super().__init__(embedder, body, targeter)

    def get_embedder(self):
        return self.embedder

    def training_step(self, train_batch, batch_idx):
        (
            label_preds,
            label_scores,
            label_logits,
            label_probas,
            attr_ohe_logits,
            attr_mhe_logits
        ) = self.forward(**train_batch)
        train_loss = self.targeter.compute_loss(
            label_logits=label_logits,
            attr_ohe_logits=attr_ohe_logits,
            attr_mhe_logits=attr_mhe_logits,
            **train_batch
        )

        logs = {"train_loss": train_loss}
        self.log("train_loss", train_loss)
        return {"loss": train_loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        (
            label_preds,
            label_scores,
            label_logits,
            label_probas,
            attr_ohe_logits,
            attr_mhe_logits
        ) = self.forward(**val_batch)

        val_loss = self.targeter.compute_loss(
            label_logits=label_logits,
            attr_ohe_logits=attr_ohe_logits,
            attr_mhe_logits=attr_mhe_logits,
            **val_batch
        )

        logs = {"val_loss": val_loss}

        predictions = self.targeter.convert_batch_outputs_to_predictions(
            predictions=label_preds,
            probas=label_probas,
            attr_ohe_logits=attr_ohe_logits,
            attr_mhe_logits=attr_mhe_logits,
            **val_batch
        )
        return {
            "loss": val_loss, "log": logs,
            "predictions": predictions
        }

    def save_valid_predictions_to_instances(
            self,
            instances: Instances,
            predictions: List[MarkupAbstract]
    ):
        return self.targeter.save_valid_predictions_to_instances(
            instances, predictions
        )

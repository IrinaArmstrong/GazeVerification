import torch

from typeguard import typechecked
from typing import (List, Tuple, Dict, Any, Union)

from gaze_verification.data_objects import (Samples, Sample, Label, Target)
from gaze_verification.modelling.training.training_model_abstract import TrainingModelAbstract
from gaze_verification.modelling.inference.inference_model_abstract import InferenceModelAbstract

# Predictor
from gaze_verification.predictors import PredictorAbstract

# Body
from gaze_verification.bodies.body_abstract import BodyAbstract

# Embedder
from gaze_verification.embedders.embedder_abstract import EmbedderAbstract


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

    Model should consist of three parts:

        - embedder - prepares input sequences and forward it through to get vector embeddings;
        - body - somehow transforms the embedder outputs (optionally);
        - predictor - makes a task-specific prediction;
    """

    """
    Need to implement: 
    from InferenceModelAbstract:
     - prepare_sample_fn
     - collate_fn
     - predict

    from TrainingModelAbstract:
     - training_step
     - validation_step
     - prediction_step - ?
     - test_step - ?
    """

    def __init__(
            self,
            embedder: EmbedderAbstract,
            body: BodyAbstract,
            predictor: PredictorAbstract
    ):
        super().__init__(embedder, body, predictor)

    def get_embedder(self):
        return self.embedder

    def _forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute logits.
        """
        embedder_output = self.embedder.forward(*args, **kwargs)
        body_output = self.body(embedder_output, *args, **kwargs)
        label_logits = self.predictor.get_logits(
            body_output, **kwargs
        )
        return label_logits

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Computes predictions, scores, predictions and logits.
        """
        label_logits = self._forward(*args, **kwargs)

        label_scores, label_probas, label_preds = self.predictor.head.predict(
            label_logits, *args, **kwargs
        )
        return (
            label_preds, label_scores,
            label_logits, label_probas,
        )

    def training_step(self, train_batch, batch_idx):
        # run embedder + body + predictor
        (
            label_preds,
            label_scores,
            label_logits,
            label_probas,
        ) = self.forward(**train_batch)

        # then compute loss in predictor
        train_loss = self.predictor.do_compute_loss(
            label_logits=label_logits,
            **train_batch
        )

        # log loss
        logs = {"train_loss": train_loss}
        self.log("train_loss", train_loss)
        return {"loss": train_loss, "log": logs}

    def validation_step(self, val_batch: Any, batch_idx: int):
        # run embedder + body + predictor
        (
            label_preds,
            label_scores,
            label_logits,
            label_probas
        ) = self.forward(**val_batch)

        # then compute loss in predictor
        val_loss = self.predictor.do_compute_loss(
            label_logits=label_logits,
            **val_batch
        )
        # log loss
        logs = {"val_loss": val_loss}

        predictions = self.predictor.convert_batch_outputs_to_predictions(
            predictions=label_preds,
            probas=label_probas,
            **val_batch
        )
        return {
            "loss": val_loss,
            "log": logs,
            "predictions": predictions
        }

    def collate_fn(self, batch: List[Sample]):
        """
        Function that takes in a batch of data Samples and puts the elements within the batch
        into a tensors with an additional outer dimension - batch size.
        The exact output type of each batch element will be a `torch.Tensor`.
        """
        # todo: write it here or cretae as passed function
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Step function called during pytorch_lightning.trainer.trainer.Trainer.predict().
        By default, it calls forward`.

        :param batch: a Current batch;
        :type batch: Any;
        :param batch_idx: an index of current batch;
        :type batch_idx: int;
        :param dataloader_idx: an index of the current dataloader;
        :type dataloader_idx: int;
        :return: predicted output;
        :rtype: Any.
        """
        # run embedder + body + predictor
        (
            label_preds,
            label_scores,
            label_logits,
            label_probas
        ) = self.forward(**batch)

        return (
            label_preds,
            label_scores,
            label_logits,
            label_probas
        )

    def predict_batch(self, batch: dict):
        with torch.no_grad():
            (
                label_preds,
                label_scores,
                label_logits,
                label_probas
            ) = self.forward(**batch)

        predictions = self.predictor.convert_batch_outputs_to_predictions(
            label_preds=label_preds,
            label_probas=label_probas,
            **batch
        )
        return predictions

    def predict(
            self,
            data: Samples,
            dataloader_kwargs: Dict[str, Any],
            device: Union[str, torch.device]
    ) -> Samples:
        """
        Custom predict for model.
        :param data: a sequence of samples with data (may not contain labels);
        :type data: Samples;
        :param dataloader_kwargs: parameters to be passed to prediction dataloader.
                Contains:
                    - prepare_sample_fn,
                    - prepare_sample_fn_kwargs,
                    - prepare_samples_fn,
                    - prepare_samples_fn_kwargs.
        :type dataloader_kwargs: dict;
        :param device: device for inference;
        :type device: Union[str, torch.device];
        :return: samples with predictions filled;
        :rtype: Samples.
        """
        predict_dataloader = self.get_dataloader(data, **dataloader_kwargs, is_predict=True)
        self.eval()

        # run inference
        output = []
        with torch.no_grad():
            for batch in predict_dataloader:
                batch_preds = self.predict_batch(
                    {k: v.to(device) for k, v in batch.items()}
                )
                output.extend(batch_preds)

        for i in range(len(data)):
            preds = output[i]
            if isinstance(preds, Target):
                data[i].predicted_label = preds
            elif isinstance(preds, list):
                # todo: may be add post-processing
                data[i].predicted_label = preds
            else:
                raise ValueError(f"Unknown target type: {type(preds)}")

        return data

    def save_valid_predictions_to_samples(
            self,
            samples: Samples,
            predictions: List[Label]
    ):
        return self.predictor.save_valid_predictions_to_instances(
            samples, predictions
        )

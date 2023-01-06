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


class ModelConfig:
    """
    Config for Model class.

    It allows to
        - set few shot learning parameters:
                * n_classes;
                * n_support;
                * n_query;
        - hold optimizer and lr_scheduler type and parameters;
    """

    def __init__(
            self,
            model_name: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            optimizer_type: str,
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_type: str,
            lr_scheduler_kwargs: Dict[str, Any]
    ):
        self.model_name = model_name
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs


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
            predictor: PredictorAbstract,
            config: ModelConfig
    ):
        super().__init__(embedder, body, predictor)
        self.config = config

    def get_embedder(self):
        return self.embedder

    def configure_optimizers(self):
        """
        Create optimizer and learning rate scheduler.
        """
        optimizer_type = getattr(torch.optim, self.config.optimizer_type)
        optimizer = optimizer_type(self.parameters(),
                                   **self.config.optimizer_kwargs)

        lr_scheduler_type = getattr(torch.optim.lr_scheduler, self.config.lr_scheduler_type)
        scheduler = lr_scheduler_type(optimizer=optimizer,
                                       **self.config.lr_scheduler_kwargs)
        scheduler = {
                'scheduler': scheduler,  # REQUIRED: The scheduler instance
                'interval': 'step',
                'frequency': 1,
                "monitor": "val_loss",
                "strict": False
            }
        return [optimizer], [scheduler]

    def _forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute logits.
        """
        embedder_output = self.embedder.forward(*args, **kwargs)
        body_output = self.body(embedder_output, *args, **kwargs)
        predictor_output = self.predictor(body_output, **kwargs)
        return predictor_output

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Computes predictions, scores, predictions and logits.
        """
        predictor_output = self._forward(*args, **kwargs)

        scores, probabilities, predictions = self.predictor.head.predict(
            predictor_output, *args, **kwargs
        )
        return (
            predictions, scores,
            predictor_output, probabilities,
        )

    def training_step(self, train_batch, batch_idx):
        # run embedder + body + predictor
        (
            predictions,
            scores,
            predictor_output,
            probabilities,
        ) = self.forward(**train_batch)

        # then compute loss in predictor
        train_loss = self.predictor.compute_loss(
            label_logits=predictor_output,
            **train_batch
        )

        # log loss
        logs = {"train_loss": train_loss}
        self.log("train_loss", train_loss)
        return {"loss": train_loss, "log": logs}

    def validation_step(self, val_batch: Any, batch_idx: int):
        # run embedder + body + predictor
        (
            predictions,
            scores,
            predictor_output,
            probabilities
        ) = self.forward(**val_batch)

        # then compute loss in predictor
        val_loss = self.predictor.compute_loss(
            label_logits=predictor_output,
            **val_batch
        )
        # log loss
        logs = {"val_loss": val_loss}

        predictions = self.predictor.convert_batch_outputs_to_predictions(
            predictions=predictions,
            probas=probabilities,
            **val_batch
        )
        return {
            "loss": val_loss,
            "log": logs,
            "predictions": predictions
        }

    def collate_fn(self, samples: List[Dict[str, Any]],
                   to_label_ids: bool = True) -> Dict[str, torch.Tensor]:
        """
        Function that takes in a batch of data Samples and puts the elements within the batch
        into a tensors with an additional outer dimension - batch size.
        The exact output type of each batch element will be a `torch.Tensor`.
        """
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        first = samples[0]
        if not isinstance(first, dict):
            raise AttributeError(
                f"Input sample to collate_fn should be type of `dict`, got: {type(first)}"
            )

        if ("label" in first.keys()) and (first['label'] is not None):
            if isinstance(first['label'], str) and to_label_ids:
                labels = [
                    self.predictor.co.get_target2idx(sample['label'])
                    for sample in samples
                ]
            elif isinstance(first['label'], torch.Tensor):
                labels = [sample['label'].item() for sample in samples]
            else:
                labels = [sample['label'] for sample in samples]
            dtype = torch.long if type(labels[0]) is int else torch.float
            batch["labels"] = torch.tensor(labels, dtype=dtype)

        # Handling of all other data keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if (k not in ("label", "label_ids") and v is not None
                    and not isinstance(v, str) and not isinstance(v, dict)):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([sample[k] for sample in samples])
                else:
                    batch[k] = torch.tensor([sample[k] for sample in samples])
        return batch

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
            predictions,
            scores,
            predictor_output,
            probabilities
        ) = self.forward(**batch)

        return (
            predictions,
            scores,
            predictor_output,
            probabilities
        )

    def predict_batch(self, batch: dict):
        with torch.no_grad():
            (
                predictions,
                scores,
                predictor_output,
                probabilities
            ) = self.forward(**batch)

        # Convert to list of Label instances
        predictions = self.predictor.convert_batch_outputs_to_predictions(
            label_preds=predictions,
            label_probas=probabilities,
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
            prediction = output[i]
            if isinstance(prediction, Label):
                data[i].predicted_label = prediction
            elif isinstance(prediction, list):
                # todo: may be add post-processing
                data[i].predicted_label = prediction
            else:
                raise ValueError(f"Unknown target type: {type(prediction)}")

        return data

    def prepare_sample_fn(self, sample: Sample, *args, **kwargs) -> Dict[str, Any]:
        """
        Contains a pipeline of transformations for preparing data from a raw Sample for model.
        The function can include various transformations,
        for example: filtering, splitting into smaller parts of data, and so on.
        """
        raise NotImplementedError


    def prepare_samples_fn(self, samples: Samples, *args, **kwargs) -> Dict[str, Any]:
        """
        Contains a pipeline of transformations for preparing data from a raw Samples sequence for model.
        The function can include various transformations,
        for example: filtering, splitting into smaller parts of data, and so on.
        """
        raise NotImplementedError

    def compute_metrics(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional measures of classification performance.
        """
        pred_labels = torch.flatten(outputs.get('predictions'))
        true_labels = torch.flatten(outputs.get('true_tags'))

        metrics = dict()

        # todo: combine base metrics and biometric (from metrics/)

        return outputs

    def log_metrics(self, loss: float = 0.0, suffix: str = '',
                    metrics_results: Dict[str, float] = None,
                    on_step: bool = False, on_epoch: bool = True):
        """
        To log metrics.
        metrics_results - dict of key = metric name, val = value of metric;
        """
        # log loss
        self.log(f"{suffix}_loss", loss, on_step=on_step,
                 on_epoch=on_epoch, prog_bar=True, logger=True)
        # log other metrics
        if metrics_results is not None:
            for key, val in metrics_results.items():
                self.log(f"{suffix}_{key}", val,
                         on_step=on_step,  # logs at this step
                         on_epoch=on_epoch,  # logs epoch accumulated metrics.
                         prog_bar=True,  # logs to the progress bar.
                         logger=True)  # logs to the logger

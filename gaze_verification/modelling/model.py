# Networks
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchmetrics

from typeguard import typechecked
from typing import (List, Tuple, Dict, Any, Union, Optional)

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
            n_support: int,
            n_query: int,
            iterations: int,
            n_classes: int,
            embedding_size: int,
            optimizer_type: str,
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_type: str,
            lr_scheduler_kwargs: Dict[str, Any]
    ):
        self.model_name = model_name
        self.n_support = n_support
        self.n_query = n_query
        self.iterations = iterations
        self.n_classes = n_classes
        self.embedding_size = embedding_size
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
            metrics: Dict[str, Any],
            config: ModelConfig
    ):
        super().__init__(embedder, body, predictor)
        self.config = config
        self.metrics = metrics

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

    def _forward(self, **kwargs) -> torch.Tensor:
        """
        Compute logits.
        """
        data = kwargs.pop("data")
        embedder_output = self.embedder.forward(data, **kwargs)
        body_output = self.body(embedder_output, **kwargs)
        predictor_output = self.predictor(body_output, **kwargs)
        return predictor_output

    def forward(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Computes predictions, scores, predictions and logits.
        """
        is_predict = kwargs.pop('is_predict', True)
        predictor_output = self._forward(**kwargs)  # gets embeddings from head
        if is_predict:
            scores, probabilities, predictions = self.predictor.head.predict(
                predictor_output, **kwargs, return_dict=False, return_dists=True
            )
            return (
                predictions,
                scores,
                predictor_output,
                probabilities,
            )
        else:
            # (loss, y_hat, target_inds, acc_val, log_p_y, dists)
            loss, predictions, true_labels, accuracy, probabilities, scores = self.predictor.head.score(
                predictor_output, return_dict=False, return_dists=True, **kwargs
            )
            return (
                loss,
                predictions,
                true_labels,
                accuracy,
                probabilities,
                scores
            )

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss and some additional metrics.
        """
        # run embedder + body + predictor embeddings ->
        # then compute loss in predictor
        (
            train_loss,
            predictions,
            true_labels,
            accuracy,
            probabilities,
            scores
        ) = self.forward(**train_batch, is_predict=False)

        # compute metrics
        metrics_results = self.compute_metrics({"predictions": predictions, "true_labels": true_labels})

        # logs metrics for each training_step
        self.log_metrics(loss=train_loss, suffix='train',
                         metrics_results=metrics_results,
                         on_step=True,
                         on_epoch=True)

        return {
            "loss": train_loss,
            "metrics_results": metrics_results
        }

    def training_epoch_end(self, outputs):
        """
        Called at the end of the training epoch
        with the outputs of all training steps.
        Here needed for ReduceLROnPlateau scheduler.
        """
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def validation_step(self, val_batch: Any, batch_idx: int):

        # run embedder + body + predictor embeddings ->
        # then compute loss in predictor
        (
            val_loss,
            predictions,
            true_labels,
            accuracy,
            probabilities,
            scores
        ) = self.forward(**val_batch, is_predict=False)

        # compute metrics
        metrics_results = self.compute_metrics({"predictions": predictions, "true_labels": true_labels})

        # logs metrics for each training_step
        self.log_metrics(loss=val_loss, suffix='val',
                         metrics_results=metrics_results,
                         on_step=True,
                         on_epoch=True)
        return {
            "loss": val_loss,
            "metrics_results": metrics_results
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
                    self.predictor.configurator.get_target2idx(sample['label'])
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
        ) = self.forward(**batch, is_predict=True)

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

    def compute_metrics(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional measures of classification performance.
        """
        pred_labels = torch.flatten(outputs.get('predictions'))
        true_labels = torch.flatten(outputs.get('true_labels'))

        metrics_results = dict()
        for metric_name, metric in self.metrics.items():
            metric_score = metric(pred_labels, true_labels)
            self._logger.info(f"{metric_name.upper()} on step = {metric_score}")
            metrics_results[metric_name] = metric_score

        return metrics_results

    def log_metrics(self,
                    loss: Union[float, torch.Tensor] = 0.0,
                    suffix: str = '',
                    metrics_results: Optional[Union[Dict[str, float], Any]] = None,
                    on_step: Optional[bool] = False,
                    on_epoch: Optional[bool] = True):
        """
        To log metrics.
        metrics_results - dict of key = metric name, val = value of metric;
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().detach().item()
        # log loss
        self.log(f"{suffix}_loss", loss,
                 on_step=on_step,
                 on_epoch=on_epoch,
                 prog_bar=True,
                 logger=True)
        # log other metrics
        if metrics_results is not None:
            for key, val in metrics_results.items():
                self.log(f"{suffix}_{key}", val,
                         on_step=on_step,  # logs at this step
                         on_epoch=on_epoch,  # logs epoch accumulated metrics.
                         prog_bar=True,  # logs to the progress bar.
                         logger=True)  # logs to the logger

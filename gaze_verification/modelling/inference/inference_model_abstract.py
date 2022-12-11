import os
import torch
from abc import abstractmethod
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Union

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.datasets import SamplesDataset

class InferenceModelAbstract(torch.nn.Module):
    """
    Abstract class for all inference-ready models.
    which can run in predict-mode to get predictions
    in Formatter-defined format.
    """
    def __init__(self, embedder, body, targeter):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

    def get_dataloader(
            self,
            samples: Samples,
            *args,
            is_predict: bool = False,
            cache_samples: bool = False,
            **kwargs
    ) -> DataLoader:
        """
        Method prepares a dataloader using custom SamplesDataset.

        :param samples: data samples;
        :type samples: Samples;

        :param is_predict: whether model is running in predict mode, defaults to False
        :type is_predict: bool

        :param cache_samples: whether to cache the results of prepare_sample_fn, defaults to False
        :type cache_samples: bool

        :return: a dataloader.
        :rtype: DataLoader
        """
        return DataLoader(
            SamplesDataset(
                self.prepare_sample_fn,
                samples,
                is_predict=is_predict,
                cache_samples=cache_samples
            ),
            *args,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def get_embedder(self):
        return self

    @abstractmethod
    def prepare_sample_fn(self, sample: Sample, is_predict: bool) -> Dict[str, Any]:
        """
        Contains a pipeline of transformations for preparing data from a raw Sample for model.
        The function can include various transformations,
        for example: filtering, splitting into smaller parts of data, and so on.
        """
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Describes a logic for collecting samples into a single batch inside DataLoader.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            data: Samples,
            dataloader_kwargs: Dict[str,Any],
            device: Union[str, torch.device]
    ) -> Samples:
        """
        Runs prediction pipeline and returns Samples
        with filled predictions related fields filled.
        :param data: data samples;
        :type data: Samples;
        :param dataloader_kwargs: some arguments for Pytorch Dataloader;
        :type dataloader_kwargs: dict;
        :param device: a device for samples storage;
        :type device: str, torch.device;
        :return: samples with filled predictions related fields filled;
        :rtype: Samples.
        """
        raise NotImplementedError

    def save_weights(self, path: str, filename: str = "model.ckpt") -> None:
        """
        Save the model's state_dict into the path folder.
        :param path: path to save a model to.
        :type path: str
        :param filename: name of the file, defaults to "model.ckpt"
        :type filename: str
        """
        os.makedirs(path, exist_ok=True)
        joint_path = os.path.join(path, filename)
        self._logger.info(f"The checkpoint will be save to {joint_path}.")
        torch.save(self.state_dict(), joint_path)

    def load_weights(self, path: str,
                     filename: Optional[str] = "model.ckpt",
                     location: str = 'cpu') -> None:
        """
        Load the model's state_dict from the path folder.
        :param path: path to load a model from (folder or file);
        :type path: str;
        :param filename: name of the file (ignored when 'path' is a path to the file),
            defaults to "model.ckpt"
        :type filename: Optional[str];
        :param location: a device name to store loaded model;
        :type location: str.
        """
        if os.path.isfile(path):
            joint_path = path
        else:
            joint_path = os.path.join(path, filename)
        self._logger.info(f"The weights will be loaded from {joint_path} to '{location}'.")
        self.load_state_dict(torch.load(joint_path, map_location=torch.device(location)))

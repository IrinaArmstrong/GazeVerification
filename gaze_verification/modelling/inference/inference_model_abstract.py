import os
import torch
from abc import abstractmethod
from torch.utils.data import DataLoader
from auto_ner import Instances, Instance
from auto_ner.core.utils import init_logger
from typing import Dict, Any, List, Optional, Union
from gaze_verification.data_processors.datasets import SamplesDataset


class InferenceModelAbstract(torch.nn.Module):
    """
    Abstract class for all inference-ready models.
    which can run in predict-mode.
    """
    def __init__(self):
        super().__init__()
        self._logger = init_logger(logger_name=self.__class__.__name__)

    def custom_dataloader(
            self,
            instances: Instances,
            *args,
            is_predict: bool = False,
            cache_samples: bool = False,
            **kwargs
    ) -> DataLoader:
        """
        Method prepares a dataloader using InstanceDataset.

        :param instances: data samples.
        :type instances: Instances

        :param is_predict: whether model is running in predict mode, defaults to False
        :type is_predict: bool

        :param cache_samples: whether to cache the results of prepare_sample_fn, defaults to False
        :type cache_samples: bool

        :return: a dataloader.
        :rtype: DataLoader
        """
        return DataLoader(
            InstanceDataset(
                self.prepare_sample_fn,
                instances,
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
    def prepare_sample_fn(self, instance: Instance, is_predict: bool) -> Dict[str, Any]:
        """
        Describe logic for preparing a sample from an Instance for your model
        """
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Describe logic for collecting samples a one batch inside DataLoader
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            data: Instances,
            dataloader_kwargs: Union[dict, DictConfig],
            device: Union[str, torch.device]
    ) -> Instances:
        raise NotImplementedError

    def save_checkpoint(self, path: str, filename: str = "model.ckpt") -> None:
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

    def load_checkpoint(self, path: str, filename: Optional[str] = "model.ckpt") -> None:
        """
        Load the model's state_dict from the path folder.

        :param path: path to load a model from (folder or file).
        :type path: str

        :param filename: name of the file (ignored when 'path' is a path to the file),
            defaults to "model.ckpt"
        :type filename: Optional[str]
        """
        if os.path.isfile(path):
            joint_path = path
        else:
            joint_path = os.path.join(path, filename)
        self._logger.info(f"The checkpoint will be loaded from {joint_path}.")
        self.load_state_dict(torch.load(joint_path, map_location=torch.device("cpu")))

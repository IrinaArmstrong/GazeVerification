import json
import numpy as np
from typeguard import typechecked
from typing import List, Any, Dict, Union, Set, Tuple

from gaze_verification.data_utils.sample import Samples
from gaze_verification.logging_handler import get_logger
from gaze_verification.target_splitters.target_splitter_abstract import TargetSplitterAbstract


@typechecked
class TargetConfigurator:
    """
    Manages the configuration of targets within the experiment.
    Based on the input file 'targets_config.json' with the configuration of classes,
    it enables mapping of the target to its index (target2idx).
    Also filters the input data by predefined splits dataset (train/validation/test).
    """

    def __init__(self, targets_config_path: str):

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        # Load config
        self.targets_config_path = targets_config_path
        with open(self.targets_config_path, encoding="utf-8") as f:
            self.original_json = json.load(f)

        self.targets = self._convert_json_to_dict(self.original_json)
        # Mappings
        self.idx2target = self._get_idx2target(self.targets)
        self.target2idx = self._get_target2idx(self.idx2target)
        self.datatype2idx, self.datatype2names = self._get_datatypes_mappings(self.targets,
                                                                              self.target2idx)
        # Encodings
        self.target2idx_ohe = self._create_one_hot_encoding(self.entities_ohe)

    def _convert_json_to_dict(
            self,
            targets_json: Dict[str, dict]
    ) -> Dict[str, dict]:

        """
        Converts input json into dict with information about targets expected to appear in data.
        So then skip targets with attribute 'skip' set to true.

        :param targets_json:
            JSON structure:
            {...,
            'Участник #1': {
                "dataset_type": 'train'
                'skip': False
                }
             ...
             }

        :return: a dict:

        targets_dict: {
            "Участник #1": {
                "idx": 4,
                "attributes": {
                    ...,
                    "dataset_type": 'train',
                    ...,
                    }
                }
            }
        """
        targets_dict = dict()
        target_idx = -1

        # extract datasets that exists in targets config
        self.dataset_type_names = TargetConfigurator._extract_datasets_type_names(targets_json)

        for target_name, target_attrs in targets_json.items():
            to_skip = target_attrs.pop("skip")
            if not to_skip:
                target_idx += 1
                targets_dict[target_name] = {
                    "idx": target_idx,
                    "attributes": target_attrs,
                }
        return targets_dict

    @staticmethod
    def _extract_datasets_type_names(targets: Dict[str, Any]) -> Set[str]:
        """
        Extract from provided data dataset type names.
        :param targets: a dict with structure:
                        {...,
                        'Участник #1': {
                            "dataset_type": 'train'
                            'skip': False
                            }
                         ...
                        }
        :type targets: a dict
        :return: a set of dataset's type names,
        :rtype: a set.
        """
        dataset_type_names = set()
        for target_name, target_attrs in targets.items():
            dataset_type = target_attrs.get("dataset_type", None)
            if dataset_type is None:
                raise AttributeError(f"Dataset type name is not set for the target, "
                                     "but it is required for correct experiment configuration.")
            dataset_type_names.add(dataset_type)
        return dataset_type_names

    @staticmethod
    def _get_idx2target(
            targets: Dict[str, Dict[str, Union[int, list, None]]]
    ) -> List[str]:
        """
        Created a mapping of target's index to target itself.
        :param targets: a dict of targets with structure:
                         {
                         "Участник #1": {
                            "idx": 4,
                            ...
                            },
                        ...
                        }
        :type targets: a dict, where a key is a target's name,
        :return: a mapping of target's index to target's name,
        :rtype: a dict.
        """
        idx2target = ["" for _ in targets]
        for target_name in targets.keys():
            idx2target[targets[target_name]["idx"]] = target_name
        return idx2target

    @staticmethod
    def _get_target2idx(
            idx2target: List[str]
    ) -> Dict[str, int]:
        """
        Created a mapping of target's name to target's index
        with respect to wise-versa mapping (index to name).
        :param idx2target: a list of target's names,
        :type idx2target: a list of str,
        :return: a mapping of target's name to target's index,
        :rtype: a dict.
        """
        target2index = {target_name: idx for idx, target_name in enumerate(idx2target)}
        return target2index

    def _create_one_hot_encoding(self, targets: List[Union[int, str]]) -> np.ndarray:
        """
        Created inner mapping for one-hot encoding of targets.
        :param targets:
        :type targets:
        :return:
        :rtype:
        """
        pass

    def _get_datatypes_mappings(self, targets: Dict[str, Union[str, bool, dict]],
                                target2idx: Dict[str, int]) -> Tuple[Dict[str, List[int]],
                                                                     Dict[str, List[str]]]:
        """
        Creates two mappings between dataset type (train, validation, test, etc.)
        and target's that belongs to it with their:
            1) indexes,
            2) names,
        :param targets: a dict-like targets, where key is target's name.
                        {
                            "Участник #1": {
                                "idx": 4,
                                "attributes": {
                                    ...,
                                    "dataset_type": 'train',
                                    ...,
                                    }
                                }
                        }
        :type targets: a dict,
        :param target2idx: a mapping of target's name to target's index,
        :type target2idx: a dict, where key is target's name,
        :return: mappings between dataset type and targets:
            1) indexes,
            2) names,
        :rtype: two dicts, where keys are dataset type.
        """
        dataset_type_to_idx = dict.fromkeys(self.dataset_type_names)
        dataset_type_to_name = dict.fromkeys(self.dataset_type_names)
        # by default all are empty lists
        for default_dataset_type in self.dataset_type_names:
            dataset_type_to_idx[default_dataset_type] = list()
            dataset_type_to_name[default_dataset_type] = list()

        for target_name, target_attrs in targets.items():
            targets_dataset_type = target_attrs.get("dataset_type")
            dataset_type_to_idx[targets_dataset_type].append(target2idx[target_name])
            dataset_type_to_name[targets_dataset_type].append(target_name)
        return dataset_type_to_idx, dataset_type_to_name

    def create_one_hot_encoding(self, targets: List[Union[int, str]],
                                idx2target: List[str]) -> np.ndarray:
        """
        Encode targets as a one-hot numeric array.
        :param targets:
        :type targets:
        :param idx2target:
        :type idx2target:
        :return:
        :rtype:
        """
        pass

    def get_idx2target(self, idx: int) -> str:
        """
        Outputs target's name based on it's index.
        :param idx: a target's index,
        :type idx: integer,
        :return: a target's name,
        :rtype: str
        """
        if idx > len(self.idx2target):
            raise ValueError(
                f"Requested target's index: `{idx}` is out of bounds ",
                f"for existing targets list (of size {len(self.idx2target)}). ",
                f"Check settings and try again."
            )
        return self.idx2target[idx]

    def get_target2idx(self, target_name: str) -> int:
        """
        Outputs target's index based on it's name.
        :param target_name: a target's name,
        :type target_name: str,
        :return: a target's index,
        :rtype: integer
        """
        if target_name not in self.target2idx.keys():
            raise ValueError(
                f"Requested target's name: `{target_name}` is not in existing targets list.",
                f"Check settings and try again."
            )
        return self.target2idx[target_name]

    def get_dataset2idx(self, dataset_type: str) -> List[int]:
        """
        Outputs all dataset's target's indexes based on dataset's type.
        :param dataset_type: a dataset's type (train, validation, test, etc.),
        :type dataset_type: str
        :return: all dataset's target's indexes,
        :rtype: a list of integer indexes.
        """
        if dataset_type in self.dataset_type_names:
            return self.datatype2idx.get(dataset_type)
        else:
            raise ValueError(f"Can not define dataset type parameter: {dataset_type}.\n"
                             f"This type is not in supported dataset types: {self.dataset_type_names}",
                             f"Skipping dataset type for target.")

    def split_samples(self, data: Samples) -> Dict[str, Samples]:
        """
        Split and (optionally) shuffle samples based on defined targets split
        """
        # get list of target values (per sample)
        targets = TargetSplitterAbstract.extract_targets(data=data)
        samples_split = TargetSplitterAbstract._split_samples(data=data,
                                                              targets=targets,
                                                              targets_split=self.datatype2names)
        return samples_split

import json
import warnings
from typing import List, Any, Dict, Union, Set

import numpy as np

from gaze_verification.logging_handler import get_logger


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
        self.datatype2idx = self._get_datatype2idx(self.targets)
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
                    'in_train': True,
                    'in_validation': False,
                    'in_test': False,
                    }
                }
            }
        """
        targets_dict = dict()
        target_idx = -1

        # extract datasets that exists in targets config
        self.dataset_type_names = TargetConfigurator._extract_datasets_type_names(targets_json)

        for target_name, target_attrs in targets_json.items():
            to_skip = target_attrs["skip"]
            if not to_skip:
                target_idx += 1
                targets_dict[target_name] = {
                    "idx": target_idx,
                    "attributes": TargetConfigurator._configure_dataset_type(target_attrs),
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
        return set([target_attrs.get("dataset_type") for _, target_attrs in targets.items()])

    def _configure_dataset_type(self, target: Dict[str, Any]) -> dict:
        """
        Converts string representation of dataset type for target (to which it belongs)
        to boolean flags dict.
        Currently supported types: c.
        Other dataset types labels will be skipped.
        :param target: a dict with structure:
                        {
                        ...,
                        "dataset_type": 'train',
                        ...
                        }
        :type target: a dict
        :return: a dict with structure:
                        {
                        'in_train': True,
                        'in_validation': False,
                        'in_test': False,
                        }
        :rtype: a dict
        """
        dataset_type_dict = dict.fromkeys(self.dataset_type_names)
        # by default all set to False
        for default_dataset_type in self.dataset_type_names:
            dataset_type_dict[default_dataset_type] = False

        dataset_type = target.get("dataset_type", None)
        if dataset_type in self.dataset_type_namesS:
            dataset_type_dict[dataset_type] = True
        elif dataset_type is not None:
            warnings.warn(f"Can not define dataset type parameter for target: {target.get('name')}.\n"
                          f"{dataset_type} not in supported dataset types: {self.dataset_type_names}"
                          f"Skipping dataset type for target.")
            # by default all set to False, so this target
            # will be skipped during datasets splitting and creation
        # If dataset type is not set (= None), just return
        return dataset_type_dict

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

    def _get_datatype2idx(self, targets: Dict[str, Union[str, bool, dict]],
                          target2idx: Dict[str, int]) -> Dict[str, List[int]]:
        """
        Creates mapping between dataset type (train, validation, test, etc.)
        and target's indexes, that belongs to it.
        :param targets: a dict-like targets, where key is target's name.
                        {
                        'in_train': True,
                        'in_validation': False,
                        'in_test': False,
                        }
        :type targets: a dict,
        :param target2idx: a mapping of target's name to target's index,
        :type target2idx: a dict, where key is target's name,
        :return: mapping between dataset type and targets,
        :rtype: a dict, where key is dataset type.
        """
        dataset_type_dict = dict.fromkeys(TargetConfigurator.DATASET_TYPES)
        # by default all are empty lists
        for type in TargetConfigurator.DATASET_TYPES:
            dataset_type_dict[type] = list()

        for target_name, target in targets.items():
            for type in TargetConfigurator.DATASET_TYPES:
                if target.get("in_" + type, False):
                    dataset_type_dict.append(target2idx[target_name])
        return dataset_type_dict

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

    def get_idx2target(self, idx: int) -> int:
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
        if dataset_type in TargetConfigurator.DATASET_TYPES:
            return self.datatype2idx.get(dataset_type)
        else:
            raise ValueError(f"Can not define dataset type parameter: {dataset_type}.\n"
                             f"This type is not in supported dataset types: {TargetConfigurator.DATASET_TYPES}",
                             f"Skipping dataset type for target.")

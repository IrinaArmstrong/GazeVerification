import json
import warnings
from typing import List, Tuple, Dict, Union

import numpy as np

from gaze_verification.logging_handler import get_logger


class TargetConfigurator:
    """

    """
    DATASET_TYPES = ['train', 'validation', 'test']

    def __init__(self, entity_config_path: str):

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

        self.entity_config_path = entity_config_path
        with open(self.entity_config_path, encoding="utf-8") as f:
            self.original_json = json.load(f)

        self.targets = self._convert_json_to_dict(self.original_json)
        # Mappings
        self.idx2target = self._get_idx2target(self.targets)
        self.target2idx = self._get_target2idx(self.idx2target)
        self.datatype2idx = self._get_datatype2idx(self.targets)
        # Encodings
        self.target2idx_ohe = self._create_one_hot_encoding(self.entities_ohe)

    @staticmethod
    def _convert_json_to_dict(
            targets_json: List[Dict[str, Union[str, bool, list]]]
    ) -> Dict[str, Dict[str, Union[int, dict, bool, str]]]:

        """
        Converts input json into dict with information about targets expected to appear in data.

        :param targets_json:
            JSON structure:
            [...,
             {'name': 'Участник #1',
             "dataset_type": 'train'
             'skip': False}
             ...]

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

        for target in targets_json:
            to_skip = target["skip"]
            if not to_skip:
                target_name = to_skip["name"]
                target_idx += 1
                targets_dict[target_name] = {
                    "idx": target_idx,
                    "attributes": TargetConfigurator._configure_dataset_type(target),
                }
        return targets_dict

    @staticmethod
    def _configure_dataset_type(target: Dict[str, Union[str, bool, List[dict]]]) -> dict:
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
        dataset_type_dict = dict.fromkeys(TargetConfigurator.DATASET_TYPES)
        # by default all set to False
        for type in TargetConfigurator.DATASET_TYPES:
            dataset_type_dict[type] = False

        dataset_type = target.get("dataset_type", None)
        if dataset_type in TargetConfigurator.DATASET_TYPES:
            dataset_type_dict[dataset_type] = True
        elif dataset_type is not None:
            warnings.warn(f"Can not define dataset type parameter for target: {target.get('name')}.\n"
                          f"{dataset_type} not in supported dataset types: {TargetConfigurator.DATASET_TYPES}",
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

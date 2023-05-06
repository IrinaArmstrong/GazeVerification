import numpy as np
from itertools import compress
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing import List, Dict, Union, TypeVar, Optional

from gaze_verification.logging_handler import get_logger
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_objects.target import ClassificationTarget

# Assumes that targets labels can be anything: str, int, etc.
TargetLabelType = TypeVar("TargetLabelType")


@typechecked
class TargetSplitterAbstract(AlgorithmAbstract, ABC):
    """
    Abstract class for reading input data in arbitrary formats into Instances.
    """

    def __init__(self, *args, is_random: bool = True, seed: int = None, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.is_random = is_random
        self.seed = seed

    @staticmethod
    def extract_targets(data: Samples,
                        target_field: Optional[str] = None,
                        split_by_name: Optional[bool] = False) -> List[TargetLabelType]:
        """
        Extracts targets from the samples
        :param data: Samples
        """
        target_field = 'label' if target_field is None else target_field
        if not hasattr(data[0], target_field):
            raise AttributeError(f"Requested for targets extraction field name: `{target_field}` "
                                 "do not exists in provided samples!")
        if isinstance(getattr(data[0], target_field), ClassificationTarget):
            if split_by_name:
                targets = [str(getattr(sample, target_field).name) for sample in data]
            else:
                targets = [str(getattr(sample, target_field).id) for sample in data]
        # Currently no support for other kinds of Targets
        else:
            targets = [getattr(sample, target_field) for sample in data]
        return targets

    @staticmethod
    def targets_split_check(targets: List[TargetLabelType],
                            targets_split: Dict[str, List[TargetLabelType]]):
        """
        Check data types consistency of targets splits and targets.
        """
        if not (isinstance(targets, np.ndarray) or isinstance(targets, list) or isinstance(targets, tuple)):
            raise AttributeError(f"Provided targets should a type of `np.ndarray`, or array-like: `list` or `tuple`, ",
                                 f" provided parameter is of type: {type(targets)}")
        # Data types check
        targets_dtype = type(targets[0])
        for split_name, split_targets in targets_split.items():
            if not isinstance(split_targets[0], targets_dtype):
                raise Exception(
                    f"Target split `{split_name}` has targets of type: {type(split_targets[0])}, "
                    f"which is not comparable with targets data type: {targets_dtype}"
                )

    @staticmethod
    def _split_samples(data: Samples,
                       targets: List[TargetLabelType],
                       targets_split: Dict[str, List[TargetLabelType]],
                       ) -> Dict[str, Samples]:
        """
        Split and (optionally) shuffle samples based on defined targets split
        """
        TargetSplitterAbstract.targets_split_check(targets, targets_split)
        result = dict()
        print("target type:", type(targets[0]))
        for split_name, split_targets in targets_split.items():
            print(f"split_targets: {split_targets}")
            mask = list(map(lambda x: x in split_targets, targets))
            masked_targets = list(compress(data, mask))
            if len(masked_targets):
                result[split_name] = Samples(list(compress(data, mask)))
        return result

    def _log_split_result(self, result: Dict[str, Samples]):
        """
        Logging slitted dataset distribution and proportions.
        :param result: splitted samples,
        :type result: dict, where a key is a dataset type naming,
        """
        message = "[_log_split_result] the dataset is divided into the following groups:"
        first_iter = True
        for name, samples in result.items():
            if first_iter:
                first_iter = False
            else:
                message += ", "
            message += f"{name} - {len(samples)} samples"
            unique_targets = np.unique([sample.user_id for sample in samples])
            message += f" with {len(unique_targets)} unique users"

        self._logger.info(message + ".")

    @abstractmethod
    def run(self, data: Union[List[Sample], Samples], **kwargs) -> Dict[str, dict]:
        """
        Run splitter for datasets based on selected implementation.
        :param data: samples of data for estimation of session's quantity.
        :type data: Samples or list of separate Samples,
        :return: target's cofiguration enriched with `dataset_type` fields,
        :rtype: dict.
        """
        raise NotImplementedError

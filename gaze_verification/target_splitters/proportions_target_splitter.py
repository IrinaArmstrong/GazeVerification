
import numpy as np
from typeguard import typechecked
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import List, Dict, Union, Optional, Tuple

from gaze_verification.data_utils.sample import Sample, Samples
from gaze_verification.target_splitters.target_splitter_abstract import TargetSplitterAbstract, TargetScheme

@typechecked
class PersonalizedTargetSplitter(TargetSplitterAbstract):
    """
    Implements splitting schema from [1]:
        Divide the entire data set by participant as follows - 25% into the test sample,
        another 25% into the validation sample and the remaining 50% into the training sample.
        This does not take into account the time period of recording sessions.

    So, there are three folds with the strict separation by users:
         - train set
         - validation set
         - test set
    """

    def __init__(
            self,
            validation_split_size: Union[float, int],
            test_split_size: Union[float, int],
            seed: int = None,
    ):
        """
        :param validation_split_size: The proportion of the validation set, the value must be
            in range (0, 1) or between [1 and len(participants)].
            In first case the ratio of total number of participants would be chosen, 
            otherwise the exact number of participants would be taken.
        :param test_split_size: The proportion of the test set, the value must be
            in range (0, 1) or between [1 and len(participants)].
            In first case the ratio of total number of participants would be chosen, 
            otherwise the exact number of participants would be taken.
        :param seed: random state
        """
        super().__init__()
        self.validation_split_size = validation_split_size
        self.test_split_size = test_split_size
        self.seed = seed

    @staticmethod
    def check_selected_splits_sizes(
            validation_split_size: Union[float, int],
            test_split_size: Union[float, int],
            targets: List[str],
            min_targets_per_split: int = 2
    ):
        """
        Checks whether selected splits sized are valid
        based on total number of unique participants in dataset.
        """
        counter = Counter(targets)
        num_unique_targets = len(counter)

    @staticmethod
    def _prepare_splits(splits: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Re-estimates splits sized to sum to 1.
        :param splits: splits ratios in form:
                    [("train", 0.7), ("test", 0.2), ("validation", 0.1)]
        :type splits: a list of tuples: (split_name, split_ratio),
        :return: corrected splits ratios,
        :rtype: a list of tuples: (split_name, split_ratio)
        """
        proportion_sum = 0
        for _, proportion in splits:
            proportion_sum += proportion
        result = list()
        for name, proportion in splits:
            result.append((name, proportion / proportion_sum))
        return result

    @staticmethod
    def check_selected_splits_ratios(
            validation_split_size: Union[float, int],
            test_split_size: Union[float, int],
            targets: List[str],
            min_targets_per_split: int = 2
    ):
        """
        Checks whether selected splits sized are valid
        based on total number of unique participants in dataset.
        """
        counter = Counter(targets)
        num_unique_targets = len(counter)

        k, v = min(counter.items(), key=lambda x: x[1])
        if v < splits_min:
            idx = y_multiclass.index(k)
            label = labels[idx]
            raise ValueError(
                f"The least populated label is {label}, it has only {v} members, which is too few. "
                f"The minimum number of groups for any class cannot be less than {splits_min}."
            )

    def run(self, data: Samples) -> Dict[str, Samples]:
        """
        Splits samples into train, validation & test sets.
        :param data: samples to split
        :return: dictionary where are the key is name of the fold and value is data
        """
        # get list of target values (per sample)
        targets = self.extract_targets(data=data)

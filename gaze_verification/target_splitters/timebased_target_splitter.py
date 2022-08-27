import random
import numpy as np
from itertools import compress
from typeguard import typechecked
from collections import Counter, OrderedDict, defaultdict
from typing import List, Dict, Union, Tuple

from gaze_verification.data_utils.sample import Samples
from gaze_verification.target_splitters.target_splitter_abstract import TargetSplitterAbstract


@typechecked
class TimebasedTargetSplitter(TargetSplitterAbstract):
    """
    Implements splitting schema from [2]:
        Divide the whole dataset by participants in a similar way to the previous method
        (on train/validation and separate test).
        At the stage of selecting users for the test dataset, each user is required to have at least 2 records
        with a time difference of at least a predefined value `T`.
        Such time-differentiated recordings will be subsequently divided into template and authentication data samples,
        as follows: authentication records should be collected severely later then template records.

    So, there are exactly THREE folds with the strict separation by users & time periods:
        - train
        - validation
        - test: template + authentication data samples

    [2] Lohr, D.J., & Komogortsev, O.V. (2022). Eye Know You Too: A DenseNet Architecture
        for End-to-end Eye Movement Biometrics.
    """

    def __init__(
            self,
            splits_proportions: List[Tuple[str, Union[float, int]]],
            seed: int = None,
            min_targets_per_split: int = 1,
            min_period_between_splits: int = 1
    ):
        """
        :param splits_proportions: The proportions of splits sets in form of:
                    [("train", 0.7), ("test", 0.2), ("validation", 0.1)]
            Split proportion value must be
            in range (0, 1) or between [1 and len(participants)].
            In first case the ratio of total number of participants would be chosen,
            otherwise the exact number of participants would be taken.
        :type splits_proportions: a list of tuples: (split_name, split_ratio),
        :param seed: seed for random state,
        :type seed: int,
        :param min_targets_per_split: a minimum number of unique targets in split,
        :type min_targets_per_split: int,
        :param min_period_between_splits: a minimum time period
                passed between the two closest in timeline recordings in splits,
                measured in ms,
        :type min_period_between_splits: timestamp in ms.
        """
        super().__init__()
        self.splits_proportions = splits_proportions
        self.seed = seed
        self.min_targets_per_split = min_targets_per_split
        self.min_period_between_splits = min_period_between_splits


    def run(self, data: Samples, **kwargs) -> Dict[str, dict]:
        """
        Run splitter for datasets based on selected implementation.
        :param data: samples of data for estimation of session's quantity.
        :type data: Samples or list of separate Samples,
        :return: target's cofiguration enriched with `dataset_type` fields,
        :rtype: dict.
        """
        pass
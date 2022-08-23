
import numpy as np
from typeguard import typechecked
from sklearn.model_selection import train_test_split
from typing import List, Dict, Union, Optional

from gaze_verification.data_utils.sample import Sample, Samples
from gaze_verification.target_splitters.target_splitter_abstract import TargetSplitterAbstract, TargetScheme

@typechecked
class StratifiedTargetSplitter(TargetSplitterAbstract):
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
            validation_ratio: float,
            test_ratio: float,
            seed: int = None,
    ):
        """
        :param validation_ratio: The proportion of the validation set, the value must be exclusively
        between 0 and 1
        :param test_ratio: The proportion of the test set, the value must be exclusively
        between 0 and 1
        :param seed: random state
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def run(self, data: Samples) -> Dict[str, Samples]:
        """
        Splits samples into train, validation & test sets.
        :param data: samples to split
        :return: dictionary where are the key is name of the fold and value is data
        """
        targets = self.extract_targets(data=data)
import random
import numpy as np
from itertools import compress
from typeguard import typechecked
from collections import Counter, OrderedDict, defaultdict
from typing import List, Dict, Union, Tuple

from gaze_verification.data_utils.sample import Samples
from gaze_verification.target_splitters.target_splitter_abstract import TargetSplitterAbstract, TargetLabelType


@typechecked
class ProportionsTargetSplitter(TargetSplitterAbstract):
    """
    Implements splitting schema from [1]:
        Divide the entire data set by participant as follows - 25% into the test sample,
        another 25% into the validation sample and the remaining 50% into the training sample.
        This does not take into account the time period of recording sessions.

    So, there are any number of folds with the strict separation by users.

    [1] Makowski, S., Prasse, P., Reich, D.R., Krakowczyk, D., Jäger, L.A., & Scheffer, T. (2021).
        DeepEyedentificationLive: Oculomotoric Biometric Identification and Presentation-Attack Detection
        Using Deep Neural Networks. IEEE Transactions on Biometrics, Behavior, and Identity Science, 3, 506-518.
    """

    def __init__(
            self,
            splits_proportions: List[Tuple[str, Union[float, int]]],
            is_random: bool = True,
            seed: int = None,
            min_targets_per_split: int = 1
    ):
        """
        :param splits_proportions: The proportions of splits sets in form of:
                    [("train", 0.7), ("test", 0.2), ("validation", 0.1)]
            Split proportion value must be
            in range (0, 1) or between [1 and len(participants)].
            In first case the ratio of total number of participants would be chosen, 
            otherwise the exact number of participants would be taken.
        :type splits_proportions: a list of tuples: (split_name, split_ratio),
        :param seed: random state
        """
        super().__init__()
        self.splits_proportions = ProportionsTargetSplitter._prepare_splits(splits_proportions)
        self.is_random = is_random
        self.min_targets_per_split = min_targets_per_split
        self.seed = seed

    @staticmethod
    def _prepare_splits(splits_proportions: List[Tuple[str, Union[float, int]]]) -> List[Tuple[str, float]]:
        """
        Re-estimates splits sized to sum to 1.
        :param splits_proportions: splits ratios in form:
                    [("train", 0.7), ("test", 0.2), ("validation", 0.1)]
        :type splits_proportions: a list of tuples: (split_name, split_ratio),
        :return: corrected splits ratios,
        :rtype: a list of tuples: (split_name, split_ratio)
        """
        proportion_sum = 0
        for _, proportion in splits_proportions:
            proportion_sum += proportion
        result = list()
        for name, proportion in splits_proportions:
            result.append((name, proportion / proportion_sum))
        return result

    @staticmethod
    def _init_check(splits_proportions: List[Tuple[str, Union[float, int]]],
                    targets: List[TargetLabelType],
                    min_targets_per_split: int
                    ):
        names_checked = set()
        proportion_sum = 0
        for name, proportion in splits_proportions:
            if proportion < 0:
                raise Exception(
                    f"Split `{name}` has proportion = {proportion}, "
                    "but it should be a positive float value."
                )
            if name in names_checked:
                raise Exception(
                    f"In the splits argument, the names of splits should not be repeated: "
                    f"repeated naming - {name}."
                )
            names_checked.add(name)
            proportion_sum += proportion
        if proportion_sum == 0:
            raise Exception(
                "In the splits argument, the sum of the proportions must not be zero."
            )
        if proportion_sum > (1 + 1e-5):
            raise Exception(
                "In the splits argument, the sum of the proportions should not exceed one.",
                f" Got: {proportion_sum}"
            )
        # Check required splits proportions (minimum unique targets number per split)
        ProportionsTargetSplitter.check_selected_splits_proportions(splits_proportions,
                                                                    targets,
                                                                    min_targets_per_split)
        # Check splits number vs. dataset size
        ProportionsTargetSplitter.splits_number_check(splits_proportions,
                                                      targets)

    @staticmethod
    def check_selected_splits_proportions(
            splits_proportions: List[Tuple[str, Union[float, int]]],
            targets: List[TargetLabelType],
            min_targets_per_split: int = 1
    ):
        """
        Checks whether the minimum selected split size is valid
        based on minimal number of targets required to present in a split.
        """
        counter = Counter(targets)
        num_unique_targets = len(counter)
        splits_num_participants = [(s_name, int(num_unique_targets * s_prop))
                                   for s_name, s_prop in splits_proportions]
        min_split_name, min_split_num = min(splits_num_participants, key=lambda x: x[1])
        if min_split_num < min_targets_per_split:
            print(splits_num_participants)
            raise ValueError(
                f"The least populated split is `{min_split_name}`, it has only {min_split_num} unique targets,"
                f" which is too few."
                f" The minimum number of targets for any split cannot be less than {min_targets_per_split}."
            )

    @staticmethod
    def splits_number_check(splits_proportions: List[Tuple[str, Union[float, int]]],
                            targets: List[TargetLabelType]):
        n_samples = len(targets)
        n_splits = len(splits_proportions)
        if n_samples < n_splits:
            raise Exception(
                f"Number of dataset samples ({n_samples}) should be not less then"
                f" number of splits ({n_splits})"
            )

    @staticmethod
    def _inter_run_check(name: str, split_group: List[TargetLabelType]):
        if len(split_group) == 0:
            raise IndexError(f"Zero elements got into the split: {name}!")

    def _ordered_targets_split(self, targets: List[str]) -> Dict[str, List[str]]:
        """
        Split targets with ordered / size-based splitting strategy`:
            N1 larger classes will belong to the first split,
            N2 next in size to the second, etc.
        :param targets: list of dataset targets
        :type targets: list of str,
        :return: splits to targets mapping,
        :rtype: dict with key - split naming.
        """
        unique_targets = np.unique(targets)
        num_unique_targets = len(unique_targets)
        num_samples_per_target = Counter(targets).most_common()

        # Select the number of targets for each split
        num_targets_per_split = OrderedDict()
        proportion_start = 0.
        for split_name, split_proportion in self.splits_proportions:
            proportion_end = proportion_start + split_proportion
            n_start = int(num_unique_targets * proportion_start)
            n_end = int(num_unique_targets * proportion_end)
            num_targets_per_split[split_name] = n_end - n_start
            proportion_start = proportion_end

        # Select exact targets for each split
        targets_per_split = defaultdict(list)
        for split_name, num_targets in num_targets_per_split.items():
            for _ in range(num_targets):
                targets_per_split[split_name].append(num_samples_per_target.pop(0)[0])
        return targets_per_split

    def _random_targets_split(self, targets: List[TargetLabelType]) -> Dict[str, List[TargetLabelType]]:
        """
        Randomly split targets.
        :param targets: list of dataset targets
        :type targets: list of str,
        :return: splits to targets mapping,
        :rtype: dict with key - split naming.
        """
        unique_targets = np.unique(targets)
        num_unique_targets = len(unique_targets)
        if self.seed is not None:  # если seed=0, считаем что random зафиксирован
            random.Random(self.seed).shuffle(unique_targets)
        else:
            random.shuffle(unique_targets)

        result = dict()
        proportion_start = 0.
        for split_name, split_proportion in self.splits_proportions:
            proportion_end = proportion_start + split_proportion
            n_start = int(num_unique_targets * proportion_start)
            n_end = int(num_unique_targets * proportion_end)
            result[split_name] = unique_targets[n_start:n_end].tolist()
            self._inter_run_check(name=split_name, split_group=result[split_name])
            proportion_start = proportion_end
        return result

    def _samples_split(self,
                       data: Samples,
                       targets: List[TargetLabelType],
                       targets_split: Dict[str, List[TargetLabelType]],
                       ) -> Dict[str, Samples]:
        """
        Split and (optionally) shuffle samples based on defined targets split
        """
        result = dict()
        for split_name, split_targets in targets_split.items():
            mask = list(map(lambda x: x in split_targets, targets))
            result[split_name] = Samples(list(compress(data, mask)))
        return result

    def run(self, data: Samples, **kwargs) -> Tuple[Dict[str, Samples], Dict[str, List[TargetLabelType]]]:
        """
        Splits samples into train, validation & test sets.
        :param data: samples to split
        :return: dictionary where are the key is name of the fold and value is data
        """
        # get list of target values (per sample)
        targets = self.extract_targets(data=data)
        ProportionsTargetSplitter._init_check(self.splits_proportions,
                                              targets,
                                              self.min_targets_per_split)
        if self.is_random:
            self._logger.info(
                f"Randomized splitting selected (seed={self.seed})"
            )
            targets_split = self._random_targets_split(targets=targets)
            samples_split = self._samples_split(data=data,
                                                targets=targets,
                                                targets_split=targets_split)
        else:
            self._logger.info("Selected ordered splitting strategy: "
                              "N1 larger classes will belong to the first split, "
                              "N2 next in size to the second, etc.")
            targets_split = self._ordered_targets_split(targets=targets)
            samples_split = self._samples_split(data=data,
                                                targets=targets,
                                                targets_split=targets_split)
        self._log_split_result(result=samples_split)
        return samples_split, targets_split

    def run_samples_split(self,
                          data: Samples,
                          targets_split: Dict[str, List[TargetLabelType]],
                          ) -> Dict[str, Samples]:
        """
        Split and (optionally) shuffle samples based on defined targets split
        """
        # get list of target values (per sample)
        targets = self.extract_targets(data=data)
        samples_split = self._samples_split(data=data,
                                            targets=targets,
                                            targets_split=targets_split)
        return samples_split

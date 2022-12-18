import torch
import random
import numpy as np
from typeguard import typechecked
from typing import List, Union
from torch.utils.data import Sampler

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects import Target


@typechecked
class FewShotBatchSampler(Sampler):
    """
    Yields a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """
    def __init__(self, targets: Union[List[int], List[Target]],
                 k_way: int, n_shot: int,
                 iterations: int = 100):
        """
        Initialize the FewShotBatchSampler object.

        :param targets: an iterable containing all the target labels for the current dataset;
        :type targets: list of targets indexes or list of Target objects;
        :param k_way: a number of random classes for each iteration;
        :type k_way: int,
        :param n_shot: a number of samples for each iteration for each class (support + query),
        :type n_shot: int,
        :param iterations: a number of iterations (episodes) per epoch,
        :type iterations: int, default set to 100.
        """
        super(FewShotBatchSampler, self).__init__(data_source=None)

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        # len(labels) == len(all_dataset) !
        if isinstance(targets[0], Target):
            self.targets = [target.id for target in targets]
            if any([target is None for target in self.targets]):
                raise AttributeError(f"All target indexes should be filled before creating Sampler!"
                                     f" In some Targets indexes are still unset.")
        else:
            self.targets = targets
        self.k_way = k_way  # number of classes per iteration
        self.n_shot = n_shot  # number of samples per class for each iteration
        self.iterations = iterations

        self._unique_classes, self._classes_counts = np.unique(self.targets, return_counts=True)  # in sorted order
        self._unique_classes = torch.LongTensor(self._unique_classes)
        self._logger.info(f"Sampler contains {self._unique_classes} unique classes")
        self._logger.info(f"Sampler contains {self._classes_counts} classes counters")

        # Create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        self._dataset_indexes = np.empty((len(self._unique_classes), max(self._classes_counts)), dtype=int) * np.nan
        self._dataset_indexes = torch.Tensor(self._dataset_indexes)

        # Count each class occurrence - store the number of samples for each class/row
        self._num_elem_per_class = torch.zeros_like(self._unique_classes)
        for idx, label in enumerate(self.targets):
            label_idx = np.argwhere(self._unique_classes == label).item()
            self._dataset_indexes[label_idx, np.where(np.isnan(self._dataset_indexes[label_idx]))[0][0]] = idx
            self._num_elem_per_class[label_idx] += 1

    def __iter__(self):
        """
        Yield a batch of indexes of samples from data.
        """
        for iteration in range(self.iterations):
            self._logger.debug(f"Prototypical sampler iteration #{iteration}")
            batch_size = self.n_shot * self.k_way
            batch = torch.LongTensor(batch_size)

            # Select classes_per_it random classes for iteration
            iter_classes_idxs = torch.randperm(len(self._unique_classes))[:self.k_way]

            for i, c in enumerate(self._unique_classes[iter_classes_idxs]):
                s = slice(i * self.n_shot, (i + 1) * self.n_shot)  # create slice
                # Get indexes of labels with current class
                label_idx = torch.arange(len(self._unique_classes)).long()[self._unique_classes == c].item()
                # Get n_shot random data samples that belongs to current class
                samples_indexes = torch.randperm(self._num_elem_per_class[label_idx])[:self.n_shot]
                if len(samples_indexes) < self.n_shot:
                    samples_indexes = random.choices(np.arange(self._num_elem_per_class[label_idx]),
                                                     k=self.n_shot)
                batch[s] = self._dataset_indexes[label_idx][samples_indexes]

            # Shuffle batch
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self) -> int:
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations
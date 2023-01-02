from typeguard import typechecked
from typing import List, Union
from gaze_verification.data_processors.datasets import FewShotBatchSampler

from gaze_verification.data_objects import Target


@typechecked
class PrototypicalBatchSampler(FewShotBatchSampler):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """
    def __init__(self, targets: Union[List[int], List[Target], list],
                 k_way: int, n_support: int, n_query: int = 0,
                 include_query: bool = False,
                 iterations: int = 100):
        """
        Initialize the PrototypicalBatchSampler object.
        :param targets: an iterable containing all the target labels for the current dataset;
        :type targets: list of targets indexes or list of Target objects;
        :param k_way: a number of random classes for each iteration;
        :type k_way: int;
        :param n_support: a number of support samples for each iteration for each class;
        :type n_support: int;
        :param n_query: a number of query samples for each iteration for each class.
                        Training objective will classify the query set
                        from seeing the support set and its corresponding labels.
                        Set by default to 0;
        :type n_query: int;
        :param include_query: If True, returns batch of size n_way * (n_support + n_query), which
                              can be split into support and query set. Simplifies
                              the implementation of sampling.
                              Set by default to False;
        :type include_query: bool;
        :param iterations: a number of iterations (episodes) per epoch;
        :type iterations: int, default set to 100.
        """
        n_shot = n_support + n_query if include_query else n_support
        super(PrototypicalBatchSampler, self).__init__(targets, k_way, n_shot, iterations)

    def __iter__(self):
        """
        Yield a batch of indexes of samples from data.
        """
        yield from super().__iter__()

    def __len__(self) -> int:
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations
from typeguard import typechecked
from typing import Callable, Any, Dict
from torch.utils.data import Dataset

from gaze_verification.data_objects.sample import Sample, Samples


@typechecked
class TrainDataset(Dataset):
    """
    Dataset with a flexible function for vectorized data preparation for few-shot learning tasks.

    :param prepare_sample_fn: function for sample preparation.
    :type prepare_sample_fn: Callable[[Samples, bool], Dict[str, Any]]

    :param samples: samples of dataset,
    :type samples: Samples,

    :param is_predict: whether model is running in predict mode, defaults to False
    :type is_predict: bool

    :param prepare_sample_fn_kwargs: parameters for sample data function preparation,
    :type prepare_sample_fn_kwargs: dict with key - parameter name, value - parameter value.
    """

    def __init__(
            self,
            prepare_sample_fn: Callable[[Samples, bool, Any], Dict[str, Any]],
            samples: Samples,
            is_predict: bool = False,
            prepare_sample_fn_kwargs: Dict[str, Any] = {}
    ):
        super(TrainDataset, self).__init__()
        self.prepare_sample_fn = prepare_sample_fn
        if not len(samples):
            raise ValueError(
                "No samples supplied to model, can't process further. "
                "Please, check if input samples list is empty."
            )
        self.samples = samples
        self.is_predict = is_predict
        self._prepare_sample_fn_kwargs = prepare_sample_fn_kwargs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.prepare_sample_fn(self.samples[idx], self.is_predict, **self._prepare_sample_fn_kwargs)
        return sample

    def __len__(self):
        return len(self.samples)
from tqdm import tqdm
from typeguard import typechecked
from typing import Callable, Any, Dict, Union
from torch.utils.data import Dataset

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects.sample import Sample, Samples


@typechecked
class SamplesDataset(Dataset):
    """
    Dataset with a flexible function for vectorized data preparation for few-shot learning tasks.
    """

    def __init__(
            self,
            samples: Samples,
            prepare_samples_fn: Callable,
            prepare_samples_fn_kwargs: Dict[str, Any] = {},
            prepare_sample_fn: Callable = None,
            prepare_sample_fn_kwargs: Dict[str, Any] = {},
    ):
        """
        :param samples: samples of dataset,
        :type samples: Samples,

        :param prepare_samples_fn: function for all dataset's samples preparation;
        :type prepare_samples_fn: Callable[[Samples, bool], Dict[str, Any]];

        :param prepare_samples_fn_kwargs: parameters for  all dataset's samples data function preparation;
        :type prepare_samples_fn_kwargs: dict with key - parameter name, value - parameter value;

        :param prepare_sample_fn: function for single sample preparation;
        :type prepare_sample_fn: Callable[[Samples, bool], Dict[str, Any]];

        :param prepare_sample_fn_kwargs: parameters for single sample data function preparation;
        :type prepare_sample_fn_kwargs: dict with key - parameter name, value - parameter value;
        """
        super(SamplesDataset, self).__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        if not len(samples):
            raise ValueError(
                "No samples supplied to model, can't process further. "
                "Please, check if input samples list is empty."
            )
        self.samples = samples
        # For full dataset
        self._prepare_samples_fn = prepare_samples_fn
        self._prepare_samples_fn_kwargs = prepare_samples_fn_kwargs
        # For single sample
        self._prepare_sample_fn_kwargs = prepare_sample_fn_kwargs
        self._prepare_sample_fn = prepare_sample_fn if prepare_sample_fn is not None else self._dummy_prepare_sample_fn

        self._prepare_samples()

    def _dummy_prepare_sample_fn(self, sample: Sample, **kwargs) -> Sample:
        """
        Dummy function for `prepare_sample_fn`,
        when nothing is required to do with single sample.
        """
        return sample

    def _prepare_samples(self):
        prepared_samples = []
        try:
            prepared_samples = self._prepare_samples_fn(self.samples,
                                                        **self._prepare_samples_fn_kwargs)
        except Exception as e:
            self._logger.error(f"Error occurred during samples preparation:\n{e}")
            self._logger.warning(f"\nNote: Samples in dataset are stored as 'unprepared'!")
        self.samples = prepared_samples
        del prepared_samples
        self._logger.info(f"Samples successfully prepared.")

    def __getitem__(self, idx) -> Union[Sample, Dict[str, Any]]:
        sample = self._prepare_sample_fn(self.samples[idx],
                                        **self._prepare_sample_fn_kwargs)
        return sample

    def __len__(self):
        return len(self.samples)
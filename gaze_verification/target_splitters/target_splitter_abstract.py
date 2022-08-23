import enum
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union

from gaze_verification.logging_handler import get_logger
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_utils.sample import Sample, Samples


@enum.unique
class TargetScheme(enum.Enum):
    """
    Provides enumeration for specific dataset splitting schemas:
    - like in [1]: Divide the entire data set by participant as follows - 25% into the test sample,
        another 25% into the validation sample and the remaining 50% into the training sample.
        This does not take into account the time period of recording sessions.
    - like in [2]: Divide the whole dataset by participants in a similar way to the previous method.
        At the stage of splitting the test sample into template and authentication records,
        take into account the time period of recording sessions - authentication records
        are collected severely lagged behind the template records by some time delta T.

    [1] Makowski, S., Prasse, P., Reich, D.R., Krakowczyk, D., Jäger, L.A., & Scheffer, T. (2021).
        DeepEyedentificationLive: Oculomotoric Biometric Identification and Presentation-Attack Detection
        Using Deep Neural Networks. IEEE Transactions on Biometrics, Behavior, and Identity Science, 3, 506-518.
    [2] Lohr, D.J., & Komogortsev, O.V. (2022). Eye Know You Too: A DenseNet Architecture
        for End-to-end Eye Movement Biometrics.
    """
    RANDOM_SPLIT = enum.auto()
    TIME_DEPENDED_SPLIT = enum.auto()

    @classmethod
    def to_str(cls):
        s = " / ".join([member for member in cls.__members__.keys()])
        return s


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

    @abstractmethod
    def run(self, samples: Union[List[Sample], Samples],
            targets_config: Dict[str, dict], **kwargs) -> Dict[str, dict]:
        """
        Run splitter for datasets based on selected implementation.
        :param samples: samples of data for estimation of session's quantity.
        :type samples: Samples or list of separate Samples,
        :param targets_config: pre-defined unique targets with additional attributes,
        :type targets_config: dict
        :return: target's cofiguration enriched with `dataset_type` fields,
        :rtype: dict.
        """
        raise NotImplementedError
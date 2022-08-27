import json
from pathlib import Path
from typeguard import typechecked
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from gaze_verification.logging_handler import get_logger
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_utils.sample import Sample, Samples
from gaze_verification.target_splitters.target_scheme import TargetScheme


@typechecked
class TargetConfigGenerator(AlgorithmAbstract):
    """
    Class for automatic generation "targets.json" (aconfiguration file for targets mappings).
    Also provides functionality for specific dataset splitting schemas:
        - like in [1]: Divide the entire data set by participant as follows - 25% into the test sample,
            another 25% into the validation sample and the remaining 50% into the training sample.
            This does not take into account the time period of recording sessions.
        - like in [2]: Divide the whole dataset by participants in a similar way to the previous method.
            At the stage of splitting the test sample into template and authentication records,
            take into account the time period of recording sessions - authentication records
            are collected severely lagged behind the template records by some time delta T.
    :param output_dir: Path to save targets configuration file,
    :type output_dir: str.

    [1] Makowski, S., Prasse, P., Reich, D.R., Krakowczyk, D., Jäger, L.A., & Scheffer, T. (2021).
        DeepEyedentificationLive: Oculomotoric Biometric Identification and Presentation-Attack Detection
        Using Deep Neural Networks. IEEE Transactions on Biometrics, Behavior, and Identity Science, 3, 506-518.
    [2] Lohr, D.J., & Komogortsev, O.V. (2022). Eye Know You Too: A DenseNet Architecture
        for End-to-end Eye Movement Biometrics.
    """

    def __init__(self, output_dir: Union[str, Path], target_scheme: Union[str, int, TargetScheme]):
        super().__init__()
        self.output_dir = output_dir
        self.target_scheme = target_scheme

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

    def set_output_dir(self, output_dir: Union[str, Path] = "./"):
        self.output_dir = output_dir

    def run(self, data: Union[List[Sample], Samples], **kwargs) -> str:
        """
        Generate targets configuration file based on input samples.

        :param data: data for targets extraction,
        :type data:  a list of Sample objects or a single Samples object,
        :return: The file name that has been generated based on provided targets.
        :rtype: str
        """
        # Extract unique targets from dataset
        unique_targets = TargetConfigGenerator.extract_targets(data)(**kwargs)

        # Separate targets & samples with selected schema into splits
        scheme = self._check_target_scheme(self.target_scheme)

        # Generate targets_config
        # Save targets_config into targets_config.json file

    def _check_target_scheme(self, scheme: Union[str, TargetScheme]) -> TargetScheme:
        """
        Check validness of target scheme selection.
        """
        if not (isinstance(scheme, str) or isinstance(scheme, TargetScheme)):
            self._logger.error(f"Provided target scheme should a type of `str` or `TargetScheme`, ",
                               f" provided parameter of type: {type(scheme)}")
            raise AttributeError(f"Provided target scheme should a type of `str` or `TargetScheme`")
        if isinstance(scheme, str):
            if scheme not in TargetScheme.get_available_names():
                self._logger.error(f"Provided target scheme should be one from available list: {TargetScheme.to_str()}",
                                   f" but was given: {scheme}")
                raise AttributeError(
                    f"Provided target scheme should be one from available list: {TargetScheme.to_str()}")
            return TargetScheme(scheme)
        return TargetScheme

    @classmethod
    def extract_targets(cls, data: Union[List[Sample], Samples]) -> Dict[str, dict]:
        """
        Collects unique target's names and their attributes (like dataset typ, etc.).

        :param data: data for targets extraction,
        :type data: a list of Sample objects or a single Samples object,

        :return: a dict of unique target's names with attributes,
                with structure:
                {...,
                 'Участник #1': {
                    'skip': False
                    }
                 ...
                 }
        """
        unique_targets = defaultdict(dict)
        for sample in data:
            target_name = sample.user_id
            if target_name is None:
                continue

            unique_targets[target_name] = {
                # TODO: generate dataset split in Target Generator!
                # "dataset_type": sample.dataset_type,
                "skip": sample.skip_sample
            }
        return unique_targets


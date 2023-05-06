import os
import json
from pathlib import Path
from typeguard import typechecked
from collections import defaultdict
from typing import List, Dict, Union, TypeVar, Optional

from gaze_verification.logging_handler import get_logger
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_objects.target import Target, ClassificationTarget
from gaze_verification.target_splitters.target_scheme import TargetScheme

# Assumes that targets labels can be anything: str, int, etc.
TargetLabelType = TypeVar("TargetLabelType")


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

    def __init__(self, output_dir: Union[str, Path],
                 target_scheme: Union[str, TargetScheme]
                 ):
        super().__init__()
        self._check_output_dir(output_dir)
        self.output_dir = output_dir
        self.target_scheme = self._check_target_scheme(target_scheme)

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

    def run(self, data: Union[List[Sample], Samples], **kwargs):  # -> Dict[str, List[TargetLabelType]]:
        """
        Generate targets configuration file based on input samples.

        :param data: data for targets extraction,
        :type data:  a list of Sample objects or a single Samples object,
        :return: generated targets configuration.
        :rtype: Dict[str, List[str]]
        """
        # Extract unique targets from dataset
        targets_config = TargetConfigGenerator.extract_targets(data,
                                                               target_field=kwargs.pop('target_field', 'user_id'))

        # Separate targets & samples with selected schema into splits
        splitter = self.target_scheme.value(**kwargs)
        _, targets_split = splitter.run(data)

        # Generate targets_config
        targets_config = self.construct_targets_config(unique_targets=targets_config,
                                                       targets_split=targets_split)
        # Save targets_config into targets_config.json file
        path_to_file = os.path.join(self.output_dir, "targets_config.json")
        with open(path_to_file, "w") as out:
            json.dump(targets_config, out, ensure_ascii=False, indent=4)
        self._logger.info(
            f"Targets config file saved to: {path_to_file}"
        )
        return targets_config

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
            self._logger.info(f"Selected target scheme name: {scheme}")
            return getattr(TargetScheme, scheme)
        return scheme

    def _check_output_dir(self, output_dir: Union[str, Path] = "./"):
        """
        Check existance of selected output directory.
        """
        output_dir = Path(output_dir).resolve()
        if not output_dir.exists():
            self._logger.warning(f"Provided output directory do not exists: {output_dir}")
            try:
                output_dir.mkdir(exist_ok=True)
                self._logger.warning(f"Output directory created by path: {output_dir}")
            except Exception as e:
                self._logger.error(f"Can't create output directory!",
                                   f"\nError occurred: {e}")
                raise FileNotFoundError(f"Cant't create output directory!")
        self._logger.info(f"Target configuration file will be saved to: {output_dir}")

    def set_output_dir(self, output_dir: Union[str, Path] = "./"):
        """ Set output directory."""
        self._check_output_dir(output_dir)
        self.output_dir = output_dir

    @classmethod
    def extract_targets(cls, data: Union[List[Sample], Samples],
                        target_field: Optional[str] = "user_id") -> Dict[TargetLabelType, dict]:
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
        if not hasattr(data[0], target_field):
            raise AttributeError(f"Requested for targets extraction field name: `{target_field}` "
                                 "do not exists in provided samples!")
        unique_targets = defaultdict(dict)
        for sample in data:
            target_name = getattr(sample, target_field)
            if target_name is None:
                continue
            elif isinstance(target_name, ClassificationTarget):
                target_name = target_name.name

            unique_targets[target_name] = {
                "skip": sample.skip_sample
            }
        return dict(unique_targets)

    def construct_targets_config(
            self,
            unique_targets: Dict[TargetLabelType, dict],
            targets_split: Dict[str, List[TargetLabelType]]
    ) -> Dict[TargetLabelType, dict]:
        """
        Constructs default targets configuration json dict.
        """
        for split_name, split_targets in targets_split.items():
            for target in split_targets:
                target_ = unique_targets.get(target)
                if target_ is None:
                    self._logger.warning(f"Target with id: {target}"
                                         f" is not in provided unique targets\n{unique_targets}")
                    continue
                target_['dataset_type'] = split_name
        return unique_targets

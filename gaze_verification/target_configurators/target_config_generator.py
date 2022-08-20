import json
from pathlib import Path
from typeguard import typechecked
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_utils.sample import Sample, Samples
from gaze_verification.target_configurators.target_configurator import TargetConfigurator


@typechecked
class TargetConfigGenerator(AlgorithmAbstract):
    """
    Class for automatic generation "targets.json" (aconfiguration file for targets mappings).

    :param output_dir: Path to save targets configuration file,
    :type output_dir: str.
    """

    def __init__(self, output_dir: Union[str, Path]):
        super().__init__()
        self.output_dir = output_dir

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
        pass

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
                    "dataset_type": 'train'
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
                "dataset_type": sample.dataset_type,
                "skip": sample.skip_sample
            }
        return unique_targets


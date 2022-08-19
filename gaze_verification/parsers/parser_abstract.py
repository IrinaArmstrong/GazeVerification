import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, List

from gaze_verification.parsers.parser_utils import is_dir_empty
from gaze_verification.algorithm_abstract import AlgorithmAbstract
from gaze_verification.data_utils.sample import Sample, Samples


class ParserAbstract(AlgorithmAbstract, ABC):
    """
    Abstract class for reading input data in arbitrary formats into Instances.
    """

    def __init__(self, *args, data_column_names: Union[str, List[str]],
                 target_column_name: str,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._data_column_names = data_column_names
        self._target_column_name = target_column_name

    @abstractmethod
    def check_emptiness(self, data_path: Union[str, List[str], Path]):
        """
        Check if the input is non-empty.

        :param data_path: path to the data file or folder.
        :type data_path: Union[str, List[str], Path]
        """
        if os.path.isfile(data_path):
            return
        elif os.path.isdir(data_path):
            if is_dir_empty(data_path):
                raise ValueError(
                    f"Class {self.__class__.__name__} ",
                    f"did not find any files in {data_path} folder",
                    f"can't process further. Please, check if input samples list is empty."
                )
            else:
                return
        else:
            raise ValueError(
                f"Class {self.__class__.__name__} "
                f"did not find a file at {data_path} and it is not a folder; "
                f"can't process further. Please, check if input samples list is empty."
            )

    def run(self, data_path: str, **kwargs) -> Samples:
        """
        Run parser for datasets for classification tasks.

        :param data_path: Path to csv file
        :type data_path: str

        :return: Transformed input file/folder into Samples/Session(s)
        :rtype: Samples
        """
        return self._process_data(data_path)

    def _process_data(self, data_path: str) -> Samples:
        raise NotImplementedError

    def _make_sample(
            self,
            guid: str,
            seq_id: int,
            label: int,
            session_id: int
    ) -> Samples:
        raise NotImplementedError

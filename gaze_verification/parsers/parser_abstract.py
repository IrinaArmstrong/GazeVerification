from abc import ABC, abstractmethod

from typing import Union, List

from auto_ner.configurators.entity_configurator import EntityConfigurator
from auto_ner.core.algorithm_abstract import AlgorithmAbstract


class ParserAbstract(AlgorithmAbstract, ABC):
    """
    Abstract class for reading input data in arbitrary formats into Instances.
    """

    SEP_ENT = EntityConfigurator.SEP_ENT
    SEP_ATTR = EntityConfigurator.SEP_ATTR
    SEP_ATTRS = EntityConfigurator.SEP_ATTRS

    @abstractmethod
    def check_emptiness(self, data: Union[str, List[str]]):
        """
        Check if the input is non-empty.

        :param data: path to the data file or folder.
        :type data: Union[str, List[str]]
        """
        raise NotImplementedError

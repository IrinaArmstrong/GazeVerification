from dataclasses import dataclass
from dataclasses_json import dataclass_json
from abc import ABC, abstractmethod

from typing import Optional, List, Tuple


@dataclass_json
@dataclass
class TargetAbstract(ABC):
    """
    Class holds target name and it's attributes.
    """
    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        return NotImplementedError


@dataclass_json
@dataclass
class Target(TargetAbstract):
    """
    General proxy class for all targets.
    """
    def __getitem__(self, item):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

@dataclass_json
@dataclass
class ClassificationTarget(Target):
    """
    Stores classification target's name and it's attributes.
    :param name: a target's name itself;
    :type name: str;
    :param id: a target's index;
    :type name: int;
    :param attributes: attributes, additional info about target
                        or, optionally, a second level of classification hierarchy;
        For example: type = "Michael", attributes = ["50 y.o.", "eye diseases"];
    :type attributes: Optional[List[str]], defaults to None;
    """
    name: str
    id: Optional[int] = None
    attributes: Optional[List[str]] = None

    @property
    def data(self) -> Tuple[str, Optional[int], Optional[List[str]]]:
        return self.type, self.id, self.attributes

    def __getitem__(self, item):
        return self.name

    def __eq__(self, other):
        return all([
            self.name == other.name,
            self.id == other.id,
            self.attributes == other.attributes  # check if list of str comparison works properly!
        ])
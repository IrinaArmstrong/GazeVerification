from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod
from gaze_verification.data_objects.target import Target, ClassificationTarget


@dataclass_json
@dataclass
class LabelAbstract(ABC):
    """
    Describes model prediction for a single sample.
    For example, it's classes probabilities, if task is multiclass classification.
    """
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


@dataclass_json
@dataclass
class Label(LabelAbstract):
    """
    General proxy class for all labels.
    """

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


@dataclass_json
@dataclass
class ClassificationLabel(Label):
    """
    Stores classification labels of a single sample.
    It is used both for training and prediction.
    The class supports the following operations:
     - Get label by index using square brackets (__getitem__).
     - Check if two labels are equal.

    :param labels: a list of available targets in a task;
    :type labels: a list of Targets objects;
    :param probabilities: a probability of the targets (for model's prediction);
    :type probabilities: Optional[Dict[str, float]], defaults to None;
    """
    labels: List[Target]
    probabilities: Optional[Dict[str, float]] = None

    def __iter__(self):
        return iter(self.labels)

    def __getitem__(self, item):
        return self.labels[item]

    def __len__(self):
        return len(self.labels)

    def __eq__(self, other):
        return all([
            self.labels == other.labels,
            self.probabilities == other.probabilities
        ])

    def __add__(self, other):
        if self.probabilities is None:
            probabilities = other.probabilities
        elif other.probabilities is None:
            probabilities = self.probabilities
        else:
            probabilities = {**self.probabilities, **other.probabilities}
        labels = []
        visited_labels = set()
        for label in self.labels + other.labels:
            if label.type not in visited_labels:
                visited_labels.add(label.type)
                labels.append(label)

        return ClassificationLabel(labels, probabilities)
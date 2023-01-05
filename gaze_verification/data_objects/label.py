from dataclasses import dataclass

import numpy as np
import torch
from dataclasses_json import dataclass_json
from typing import Optional, List, Dict, Union
from abc import ABC, abstractmethod

from gaze_verification.data_objects.target import Target
from gaze_verification.logging_handler import get_logger

logger = get_logger(
            name=__name__,
            logging_level="INFO"
        )

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
    probabilities: Optional[Dict[Union[str, int], float]] = None

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

    def __repr__(self):
        """
        Returns string representation of Label object.
        :rtype: str.
        """
        s = f"Label:"
        with_probabilities = False
        if self.probabilities is not None:
            with_probabilities = True
        for i in range(len(self.labels)):
            label = self.labels[i]
            s += f" {label}"
            if with_probabilities:
                prob = self.probabilities.get(label.name)
                if prob is None:
                    prob = self.probabilities.get(label.id)
                if prob is None:
                    prob = "<not_found>"
                s += f" ({prob})"
            s += ", "
        return s


@dataclass_json
@dataclass
class PrototypicalLabel(Label):
    """
    Stores labels of a single sample fpr Prototypical network algorithm.
    It is used both for training and prediction.
    The class supports the following operations:
     - Get label by index using square brackets (__getitem__).
     - Check if two labels are equal.

    :param labels: a list of available targets in a task;
    :type labels: a list of Targets objects;
    :param probabilities: a probability of the targets (for model's prediction);
    :type probabilities: Optional[Dict[str, float]], defaults to None;
    :param distances: a distances matrix [len(query_embeddings), len(prototypes)];
    :type distances: Optional[Union[torch.Tensor, np.ndarray]], defaults to None.
    """
    labels: List[Target]
    probabilities: Optional[Dict[Union[str, int], float]] = None
    distances: Optional[Union[torch.Tensor, np.ndarray]] = None

    def __iter__(self):
        return iter(self.labels)

    def __getitem__(self, item):
        return self.labels[item]

    def __len__(self):
        return len(self.labels)

    def __eq__(self, other):
        return all([
            self.labels == other.labels,
            self.probabilities == other.probabilities,
            self.distances == other.distances
        ])

    def __add__(self, other):
        if self.probabilities is None:
            probabilities = other.probabilities
        elif other.probabilities is None:
            probabilities = self.probabilities
        else:
            probabilities = {**self.probabilities, **other.probabilities}

        if self.distances is None:
            distances = other.distances
        elif other.distances is None:
            distances = self.distances
        else:
            try:
                distances = self.distances + other.distances
            except:
                logger.warning(f"Distance matrices are not broadcastable for addition: {self.distances.shape} "
                               f"and {other.distances.shape}.\nFinal distance matrix will be initialized with 'None'.")
                distances = None
        labels = []
        visited_labels = set()
        for label in self.labels + other.labels:
            if label.type not in visited_labels:
                visited_labels.add(label.type)
                labels.append(label)

        return PrototypicalLabel(labels, probabilities, distances)

    def __repr__(self):
        """
        Returns string representation of Label object.
        :rtype: str.
        """
        s = f"Label:"
        with_probabilities = False
        if self.probabilities is not None:
            with_probabilities = True
        for i in range(len(self.labels)):
            label = self.labels[i]
            s += f" {label}"
            if with_probabilities:
                prob = self.probabilities.get(label.name)
                if prob is None:
                    prob = self.probabilities.get(label.id)
                if prob is None:
                    prob = "<not_found>"
                s += f" ({prob})"
            s += ", "
        return s
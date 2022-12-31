# Basic
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional

# DTO
import dataclasses
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from gaze_verification.data_objects.target import Target
from gaze_verification.data_objects.label import Label

# Serialization
import json
import pickle
_HIGHEST_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Logging
from gaze_verification import logging_handler
logger = logging_handler.get_logger(__name__,
                                    log_to_file=False)


class DataInitializationException(ValueError):
    """
    Exception indicates error during data initialization
    """


@dataclass_json
@dataclass
class Sample:
    """
    Single gaze sample instance.
    """
    guid: int
    seq_id: int
    session_id: int
    label: Optional[Union[int, Target]] = None
    user_id: Optional[int] = None
    data: Optional[Union[np.ndarray, List[float]]] = None
    data_type: Optional[str] = None
    dataset_type: Optional[str] = None
    stimulus_type: Optional[str] = None
    stimulus_file: Optional[str] = None
    stimulus_data: Optional[Union[np.ndarray, List[float]]] = None
    skip_sample: Optional[bool] = False
    additional_attributes: Optional[Dict[str, Any]] = None
    # compliting after creation
    predicted_label: Optional[Union[int, Label]] = None

    @property
    def length(self):
        return len(self.data)

    def __post_init__(self):
        pass

    def __repr__(self):
        """
        Returns a representation of the object.
        """
        s = f"Sample guid={self.guid}"
        if self.label is not None:
            s += f" label={self.label}"
        if self.user_id is not None:
            s += f" user_id={self.user_id}"
        if self.session_id is not None:
            s += f" session_id={self.session_id}"
        if self.stimulus_type is not None:
            s += f" stimulus_type={self.stimulus_type}"
        if self.data_type is not None:
            s += f" data_type={self.data_type}"
        if self.dataset_type is not None:
            s += f" dataset_type={self.dataset_type}"
        if self.predicted_label is not None:
            s += f" predicted_label={self.predicted_label}"
        return s

    def __add__(self, other: 'Sample') -> 'Sample':
        """
        Adds another sample data to current.
        TODO!
        """
        pass

    def __eq__(self, other: 'Sample') -> bool:
        """
        Method is used to compare two objects
        based on their fields content.
        """
        if not isinstance(other, Sample):
            return NotImplemented("Provided object is not the same type!")
        return (
                (self.guid, self.seq_id, self.session_id, self.user_id, self.data_type) ==
                (other.guid, other.seq_id, other.session_id, other.user_id, other.data_type))

    def to_str(self) -> str:
        """
        Return string representation.
        """
        repr = dataclasses.asdict(self)
        ignore_fields = [
            field_name for field_name, field in repr.items()
            if (isinstance(field, np.ndarray)
                or isinstance(field, pd.DataFrame) or (field is None))
        ]

        _ = [repr.pop(field_name) for field_name in ignore_fields]
        return str(repr)

    @classmethod
    def load_from_str(cls, saved_params_str: str):
        """
        Load from dict of sample parameters
        passed in format of json string (convertable to dict).
        """
        try:
            saved_params = eval(saved_params_str)
        except Exception as ex:
            logger.error(f"Error converting string page parameters: {ex}")
            return None
        return cls(**saved_params)

    def update(self, new_attr: Dict[str, Any]):
        """
        Inplace updates attribute in current class instance.
        If attribute do not exists, do nothing.
        :param new_attr: dict, where key - is an attribute name, value - new value to be set;
        :type new_attr: dict, key - str, value - any type;
        """
        for key, value in new_attr.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Attribute {key} is updated.")


@dataclass_json
@dataclass
class Samples:
    """
    The class is used for intercommunication between all the components.
    it stores training samples and allows the following operations:

    - Iterate by training samples.
    - Get an sample by index using square brackets (__getitem__).
    - Get number of samples using len().
    - Concatenate objects of class Samples using operation '+'.

    :param samples: list of Samples
    :type samples: List[Samples]
    """
    samples: List[Sample]

    def __post_init__(self):
        if not self.samples:
            raise DataInitializationException(
                "Empty Samples are forbidden! Please, add at least one sample."
            )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def __add__(self, other):
        return Samples(self.samples + other.samples)

    def save(self,
             path: str,
             as_single_file: bool,
             engine: str,
             n_samples: int = -1,
             **kwargs):
        """
        Dump the dataset into json or pickle.

        :param path: path to save the data to (either file or directory)
        :type path: str

        :param as_single_file: whether to save the dataset as a single file "{path}.{engine}",
            otherwise each sample is saved into "{path}/{i}.{engine}" for i = [0, ..., n_samples)
        :type as_single_file: bool

        :param engine: the way of serialization ("json" or "pickle")
        :type engine: str

        :param n_samples: how many first samples to save; if -1,
            then all samples as saved; defaults to -1
        :type n_samples: int

        :param **kwargs: contains useful additional keys for serialization engines.
        """
        if n_samples == -1:
            n_samples = len(self)

        if as_single_file:
            samples = self if n_samples == len(self) else Samples(
                self[:n_samples])
            if engine == "json":
                content = samples.to_json()
                with open(path, "w", encoding="utf-8") as f:
                    # **kwargs defaults: {'skipkeys': False, 'ensure_ascii': True,
                    # 'allow_nan': True, 'indent': None, 'separators': None, 'sort_keys': False}
                    f.write(content, **kwargs)
            elif engine == "pickle":
                with open(path, "wb") as f:
                    pickle.dump(samples, f, **kwargs)
            else:
                raise ValueError(f"Unknown engine {engine}!")
            return

        # as directory
        for i, sample in enumerate(self[:n_samples]):
            fpath_i = os.path.join(path, f"{i}.{engine}")
            if engine == "json":
                content = sample.to_json()
                with open(fpath_i, "w", encoding="utf-8") as f:
                    # **kwargs defaults: {'skipkeys': False, 'ensure_ascii': True,
                    # 'allow_nan': True, 'indent': None, 'separators': None, 'sort_keys': False}
                    json.dump(content, f, **kwargs)
            elif engine == "pickle":
                with open(fpath_i, "wb") as f:
                    pickle.dump(sample, f, **kwargs)
            else:
                raise ValueError(f"Unknown engine {engine}!")

    def save_json(self,
                  path: str,
                  as_single_file: bool,
                  n_samples: int = -1,
                  **kwargs):
        """
        Dump the dataset into json.
        Useful **kwargs with defaults: {
            'skipkeys': False, 'ensure_ascii': True,
            'separators': None, 'sort_keys': False
        }
        """
        logger.warning(f"Only Samples without `data` or with `data` of type `int` can be saved to json!")
        return self.save(path,
                         as_single_file,
                         engine="json",
                         n_samples=n_samples)

    def save_pickle(self,
                    path: str,
                    as_single_file: bool,
                    n_samples: int = -1,
                    **kwargs):
        """
        Dump the dataset into pickle.
        Useful **kwargs with defaults: {
            'protocol': None, 'fix_imports': True
        }
        """
        return self.save(path,
                         as_single_file,
                         engine="pickle",
                         n_samples=n_samples)

    @classmethod
    def _load(cls, path: str, engine: str, n_samples: int = -1, **kwargs):
        """
        Load the dataset from json or pickle.
        """
        # Load from single file
        if os.path.isfile(path):
            if engine == "json":
                with open(path, "r", encoding="utf-8") as f:
                    return cls.from_json(f.read())
            elif engine == "pickle":
                with open(path, "rb") as f:
                    # **kwargs default: {'fix_imports': True, 'encoding': "ASCII", 'errors': "strict"}
                    return pickle.load(f, **kwargs)
            else:
                raise ValueError(f"Unknown engine {engine}!")
        # Load from folder
        elif os.path.isdir(path):
            samples = []
            filenames = sorted(
                filter(lambda x: x.endswith(f".{engine}"), os.listdir(path)),
                key=lambda x: int(os.path.splitext(x)[0])  # "12.json" -> 12
            )
            if n_samples >= 0:
                filenames = filenames[:n_samples]
            for filename in filenames:
                if engine == "json":
                    with open(os.path.join(path, filename),
                              "r",
                              encoding="utf-8") as f:
                        # TODO: check if json.load(f) works well with datclass.from_json()
                        sample = Sample.from_json(f.read())
                elif engine == "pickle":
                    with open(os.path.join(path, filename), "rb") as f:
                        # **kwargs default: {'fix_imports': True, 'encoding': "ASCII", 'errors': "strict"}
                        sample = pickle.load(f, **kwargs)
                else:
                    raise ValueError(f"Unknown engine {engine}!")

                samples.append(sample)
            return Samples(samples)

        else:
            raise FileNotFoundError(f"Could not find data at {path}!")

    @classmethod
    def load(cls, path: str, engine: str, n_samples: int = -1):
        """
        Load the dataset from json or pickle.

        :param path: path to load the data from (either file or directory)
        :type path: str

        :param engine: the way of serialization ("json" or "pickle")
        :type engine: str

        :param n_samples: how many first samples to load; if -1, then all samples as loaded;
            ignored if path is file; defaults to -1
        :type n_samples: int
        """
        try:
            return cls._load(path, engine, n_samples)
        except Exception as e:
            raise Exception(
                f"Could not load Samples due to the error. "
                f"Check if the path is correct {e}")

    @classmethod
    def load_json(cls, path: str, n_samples: int = -1):
        """
        Load the dataset from json.
        """
        return cls.load(path, engine="json", n_samples=n_samples)

    @classmethod
    def load_pickle(cls, path: str, n_samples: int = -1):
        """
        Load the dataset from pickle.
        Useful **kwargs with defaults: {
            'fix_imports': True, 'encoding': "ASCII",
            'errors': "strict"
        }
        """
        return cls.load(path, engine="pickle", n_samples=n_samples)
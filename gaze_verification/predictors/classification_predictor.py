import torch
from abc import abstractmethod, ABC
from typeguard import typechecked
from typing import Dict, Any, List, Tuple, Optional, Type

from gaze_verification.logging_handler import get_logger
from gaze_verification.data_objects import (Sample, Samples, Label, Target)
from gaze_verification.target_configurators.target_configurator import TargetConfigurator
from gaze_verification.predictors.predictor_abstract import PredictorAbstract

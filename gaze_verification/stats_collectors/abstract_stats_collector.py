# Basic
import os
import numpy as np
from typing import List, Dict, Any

# DTO
import dataclasses
from dataclasses import dataclass


@dataclass
class QuantitativeStats:
    mean: float
    std: float
    median: float
    min: float
    max: float


@dataclass
class CategoricalStats:
    number: int
    counts: Dict[float]
    min_count: float
    max_count: float
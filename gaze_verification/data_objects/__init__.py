from .sample import Sample, Samples
from .target import Target, ClassificationTarget
from .label import Label, ClassificationLabel, PrototypicalLabel
from .utils import reduce_mem_usage

__all__ = [
    "Sample",
    "Samples",
    "Label",
    "ClassificationLabel",
    "PrototypicalLabel",
    "Target",
    "ClassificationTarget",
    "reduce_mem_usage"
]
import numpy as np
from typeguard import typechecked
from collections import Counter, OrderedDict, defaultdict
from typing import List, Dict, Union, Tuple

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.segmentors.segmentation_utils import generative_sliding_window_1d
from gaze_verification.data_processors.segmentors.segmentor_abstract import SegmentorAbstract


@typechecked
class NonOverlappingSegmentor(SegmentorAbstract):
    """
        Implements following seqmentation schema from [1]:
            ...

        [1] Lohr, D.J., & Komogortsev, O.V. (2022).
            Eye Know You Too: A DenseNet Architecture for End-to-end Biometric Authentication via Eye Movements.
            ArXiv, abs/2201.02110.
        """

    def __init__(self, segment_length: int,
                 min_completness_ratio: float = 0.85,
                 fillvalue: Union[int, float] = None,
                 verbose: bool = True):
        super().__init__(verbose)
        self.segment_length = segment_length
        self.min_completness_ratio = min_completness_ratio
        self.fillvalue = fillvalue

    def build_segments(self, sample: Sample) -> List[Sample]:
        """
        Segment data sequences from Samples on M parts, each with selected length L.
        Segments do not overlap - i.e. not a sliding window, where window_size > stride.
        Here strictly: window_size == stride.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: List with N formatted Samples
        :rtype: List[Sample]
        """
        sample_data = sample.data  # os size [data_sample_length, dims]

        segmented_sample = []

        for sequence_idx, sequence in enumerate(sample_data):
            sample_segments = []
            for segment_idx, segment in enumerate(generative_sliding_window_1d(sequence, segment_length,
                                                                               segment_length, fillvalue)):
                if sum(map(lambda x: x != fillvalue, segment)) >= (min_completness_ratio * segment_length):
                    sample_segments.append(segment)
                    print(segment)

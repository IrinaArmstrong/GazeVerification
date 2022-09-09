import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from collections import Counter, OrderedDict, defaultdict
from typing import List, Dict, Union, Tuple

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.segmentors.segmentation_utils import (sequential_sliding_window_2d,
                                                                             rearrange_dimensions)
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
                 fill_value: Union[int, float] = None,
                 verbose: bool = True):
        super().__init__(verbose)
        self.segment_length = segment_length
        self.min_completness_ratio = min_completness_ratio
        self.fill_value = fill_value

    def build_segmented_dataset(self, samples: Samples) -> Samples:
        """
        Create a new dataset containing segmented and formatted Samples.

        :param samples: DataClass containing N formatted Instances
        :type samples: Instances

        :return: Samples object containing N formatted Samples
        :rtype: Samples
        """
        segmented_samples = []
        for sample in tqdm(samples, total=len(samples), desc="Segmenting dataset..."):
            try:
                segmented_samples.extend(self.build_segments(sample))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset segmentation: {e}"
                                   f"\nSkipping sample: {sample.guid}.")

        self._logger.info(f"From given dataset of {len(samples)} samples created {len(segmented_samples)}")

        # Segmented samples guids need to be re-assigned as here they appears to be not unique
        # Initial guid for future usage is stored in sample.additional_attributes
        for segment_idx, sample in enumerate(segmented_samples):
            sample.guid = segment_idx

        return Samples(segmented_samples)

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
        sample_data = sample.data  # initial of size [data_sample_length, n_dims]
        sample_data = rearrange_dimensions(sample_data, "ij->ji")  # to size [n_dims, data_sample_length]
        n_dims = sample_data.shape[0]

        segments = []

        n_complete_samples = int(np.floor(sample_data.shape[-1] / self.segment_length))
        completness_ratio = (sample_data.shape[-1] - n_complete_samples * self.segment_length) / self.segment_length

        for segment_idx, segment in enumerate(np.squeeze(sequential_sliding_window_2d(sample_data,
                                                                                      window_size=(
                                                                                              n_dims,
                                                                                              self.segment_length),
                                                                                      dx=self.segment_length,
                                                                                      dy=1), axis=0)):
            additional_attributes = sample.additional_attributes
            additional_attributes['segment_idx'] = segment_idx
            additional_attributes['initial_sample_guid'] = sample.guid
            additional_attributes['completness_ratio'] = 1  # always completed data
            segment_sample = Sample(
                guid=sample.guid,  # further need to be re-assigned as here they appears to be not unique
                seq_id=sample.seq_id,
                label=sample.label,
                session_id=sample.session_id,
                data=segment,
                data_type=sample.data_type,
                dataset_type=sample.dataset_type,
                stimulus_type=sample.stimulus_type,
                skip_sample=sample.skip_sample,
                additional_attributes=additional_attributes
            )
            segments.append(segment_sample)

        # If last segment of sample's data is completed enough - add it also
        if completness_ratio >= self.min_completness_ratio:
            existing_segment_len = (sample_data.shape[-1] - n_complete_samples * self.segment_length)
            segment = sample_data[:, -existing_segment_len:]
            segment = np.concatenate(
                (segment,
                 np.full((sample_data.shape[0], self.segment_length - existing_segment_len),
                         fill_value=self.fill_value, dtype=sample_data.dtype)),
                axis=1)

            additional_attributes = sample.additional_attributes
            additional_attributes['segment_idx'] = segment_idx + 1
            additional_attributes['initial_sample_guid'] = sample.guid
            additional_attributes['completness_ratio'] = completness_ratio
            segment_sample = Sample(
                guid=sample.guid,  # further need to be re-assigned as here they appears to be not unique
                seq_id=sample.seq_id,
                label=sample.label,
                session_id=sample.session_id,
                data=segment,
                data_type=sample.data_type,
                dataset_type=sample.dataset_type,
                stimulus_type=sample.stimulus_type,
                skip_sample=sample.skip_sample,
                additional_attributes=additional_attributes
            )
            segments.append(segment_sample)

        return segments

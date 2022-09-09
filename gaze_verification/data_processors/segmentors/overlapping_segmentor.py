import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from typing import List, Union

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.segmentors.segmentation_utils import (sequential_sliding_window_2d,
                                                                             rearrange_dimensions)
from gaze_verification.data_processors.segmentors.segmentor_abstract import SegmentorAbstract


@typechecked
class OverlappingSegmentor(SegmentorAbstract):
    """
        Implements following seqmentation schema:
            Each gaze recording is split into overlapping windows
            that intersect at some selected interval delta_t.
            Each window has length of t s. (T = t*sampling_frequency time steps)
            and is created using a N-dim rolling window.

            Excess time steps at the end of a recording that would form only a
            partial window can either be filled with a fill value up to the full window,
            or completely discarded (by setting the parameter controlling the minimum degree
            of filling of the window with real values to 0).
            see more:
            https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks
        """

    def __init__(self, segment_length: int,
                 segmentation_step: int,
                 min_completness_ratio: float = 0.85,
                 fill_value: Union[int, float] = None,
                 verbose: bool = True):
        super().__init__(verbose)
        self.segment_length = segment_length
        self.segmentation_step = segmentation_step
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
        Segments overlap as they are generated using a sliding window, where stride does not depend on window_size.
        So two sequential windows will overlap on: n_overlapped_elements =  window_size - stride.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: List with N formatted Samples
        :rtype: List[Sample]
        """
        sample_data = sample.data  # initial of size [data_sample_length, n_dims]
        sample_data = rearrange_dimensions(sample_data, "ij->ji")  # to size [n_dims, data_sample_length]
        n_dims = sample_data.shape[0]

        segments = []
        n_complete_samples = ((sample_data.shape[-1] - self.segment_length) // self.segmentation_step) + 1

        # same calculations as for `same` type of padding in convolutions
        # https://medium.com/swlh/convolutional-neural-networks-part-2-padding-and-strided-convolutions-c63c25026eaa
        expected_padding = (self.segment_length - 1) / 2
        completness_ratio = (self.segment_length - expected_padding) / self.segment_length

        for segment_idx, segment in enumerate(np.squeeze(sequential_sliding_window_2d(sample_data,
                                                                                      window_size=(
                                                                                              n_dims,
                                                                                              self.segment_length),
                                                                                      dx=self.segmentation_step,
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
            expected_padding = int(np.ceil(expected_padding)) if int(expected_padding) < 1 else int(expected_padding)
            segment = sample_data[:, -int(self.segment_length - expected_padding):]
            segment = np.concatenate(
                (segment,
                 np.full((sample_data.shape[0], expected_padding),
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
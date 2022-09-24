import unittest
from pathlib import Path

import numpy as np

from gaze_verification.data_objects.sample import Samples, Sample
from gaze_verification.data_processors.normalizer import ZScoreNormalizer


class TestZScoreNormalizer(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._init_data_path = self._current_dir / "test_data_samples" / "filtering_samples.pickle"

    def test_normalization(self):
        init_sample = Samples.load_pickle(str(self._init_data_path))
        # assuming here is single Sample in Samples
        sample_data = init_sample[0].data
        n_elements, n_dims = sample_data.shape
        axis = 0

        normalizer = ZScoreNormalizer(copy=False)
        norm_sample = normalizer.run(init_sample, axis=axis)

        # check mean to be zero
        for mu in np.mean(norm_sample[0].data, axis=axis):
            self.assertTrue(round(mu - 0.0, 4) == 0)

        # check std to be close to one
        for std in np.std(norm_sample[0].data, axis=axis):
            self.assertTrue(round(std - 1.0, 4) == 0)


if __name__ == '__main__':
    unittest.main()

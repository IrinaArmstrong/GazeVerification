import unittest
from pathlib import Path
from gaze_verification.data_objects.sample import Samples, Sample
from gaze_verification.data_processors.filters.savitzky_golay_filter_2d import SavitzkyGolayFilter2D


class TestSavitzkyGolayFilter2D(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._init_data_path = self._current_dir / "test_data_samples" / "filtering_samples.pickle"

    def test_filtration(self):
        init_sample = Samples.load_pickle(str(self._init_data_path))
        # assuming here is single Sample in Samples
        sample_data = init_sample[0].data
        n_elements, n_dims = sample_data.shape

        for window_size, order in zip([5, 7, 9, 11, 13, 15, 17, 19, 21],
                                      [2, 3, 3, 5, 5, 7, 7, 9, 9]):
            sg_filter = SavitzkyGolayFilter2D(window_size, order,
                                              derivative_type=0,  # equals to no derivative
                                              keep_erroneous_samples=False, verbose=True)
            filtered_sample = sg_filter.filter_dataset(init_sample,
                                                       dim_subscript=[(0, 2), (1, 3)])
            self.assertTupleEqual(filtered_sample[0].data.shape, (n_elements, n_dims))


if __name__ == '__main__':
    unittest.main()

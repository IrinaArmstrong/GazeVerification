import numpy as np
from tqdm import tqdm
from math import factorial
from typeguard import typechecked
from typing import List, Union, Optional

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.filters.filter_abstract import FilterAbstract
from gaze_verification.data_processors.filters.filtration_utils import Derivative


@typechecked
class SavitzkyGolayFilter1D(FilterAbstract):
    """
    The Savitzky Golay filter is a particular type of low-pass filter, well adapted for data smoothing.

    The idea of Savitzky-Golay filters is simple – for each sample in the filtered sequence,
    take its direct neighborhood of N neighbors and fit a polynomial to it.
    Then just evaluate the polynomial at its center (and the center of the neighborhood), point 0,
    and continue with the next neighborhood.

    Also Savitzky-Golay filters producing smoothened signal gradients / derivatives.
    The main point of Savitzky-Golay gradients is to compute the derivative at a given point,
    simply by computing the derivative of the fitted polynomial.

    Those filters are useful for filtering out the noise when the frequency span of the signal is large.
    They are reported to be optimal for minimizing the least-square error
    in fitting a polynomial to frames of the noisy data.

    For further information see:
        description: https://bartwronski.com/2021/11/03/study-of-smoothing-filters-savitzky-golay-filters/
        implementation details: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    def __init__(self,
                 window_size: int,
                 order: int,
                 rate: Optional[int] = 1,
                 deriviate: Optional[Union[str, int, Derivative]] = Derivative.NONE,
                 keep_erroneous_samples: bool = True,
                 verbose: bool = True):
        super().__init__(verbose)
        self.window_size = window_size
        self.order = order
        self.rate = rate
        self.derivative = self._check_derivative(deriviate)
        self.keep_erroneous_samples = keep_erroneous_samples
        self.check_selected_parameters()

    def check_selected_parameters(self):
        """
        Checks whether the selected window size and order pair is valid:
         - Window size is required to be a positive odd number, larger then 1,
         - Window size is required to be larger then selected order at least on 2
            (1 point for each window side),
        """
        try:
            self.window_size = np.abs(int(self.window_size))
            self.order = np.abs(int(self.order))
        except ValueError as e:
            raise ValueError("Window size and order have to be of type 'int'")
        if self.window_size % 2 != 1 or self.window_size <= 1:
            raise TypeError("Window size must be a positive odd number larger then 1")
        if self.window_size < self.order + 2:
            raise TypeError(f"Window size = {self.window_size} is too small for the polynomials order = {self.order}")

    def _check_derivative(self, derivative: Union[str, int, Derivative]) -> Derivative:
        """
        Check validness of derivative selection.
        """
        if not (isinstance(derivative, str) or isinstance(derivative, int) or isinstance(derivative, Derivative)):
            self._logger.error(f"Provided derivative should a type of `str`, `int` or `Derivative`, ",
                               f" provided parameter of type: {type(derivative)}")
            raise AttributeError(f"Provided derivative should a type of `str`, `int` or `Derivative`")

        if isinstance(derivative, str):
            if derivative not in Derivative.get_available_names():
                self._logger.error(f"Provided derivative should be one from available list: {Derivative.to_str()}",
                                   f" but was given: {derivative}")
                raise AttributeError(
                    f"Provided derivative should be one from available list: {Derivative.to_str()}")
            self._logger.info(f"Selected derivative: {derivative}")
            return getattr(Derivative, derivative)

        if isinstance(derivative, int):
            if derivative > len(Derivative):
                self._logger.error(f"Provided derivative type index is out of bounds for existing types. "
                                   f"It should be one from available list: {list(range(0, len(Derivative)))}"
                                   f" but was given: {derivative}")
                raise AttributeError(
                    f"Provided derivative type index is out of bounds for existing types. "
                    f"It should be one from available list: {list(range(1, len(Derivative) + 1))}"
                    f" but was given: {derivative}")
            self._logger.info(f"Selected derivative: {derivative}")
            return Derivative(derivative)
        return derivative

    def filter_dataset(self, samples: Samples) -> Samples:
        """
        Create a new dataset containing filtered Samples.

        :param samples: DataClass containing N filtered Samples
        :type samples: Instances

        :return: Samples object containing N filtered Samples
        :rtype: Samples
        """
        segmented_samples = []
        for sample in tqdm(samples, total=len(samples), desc="Filtering dataset..."):
            try:
                segmented_samples.append(self.filter_sample(sample))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset filtration: {e}"
                                   f"\nSkipping sample: {sample.guid}.")
                # keep sample even if it's data is not filtered
                if self.keep_erroneous_samples:
                    segmented_samples.append(sample)

        return Samples(segmented_samples)

    def filter_sample(self, sample: Sample) -> Sample:
        """
        Filter data sequences from Samples according to predefined logic.

        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :return: Sample object with filtered data field,
        :rtype: Sample
        """
        sample_data = sample.data  # initial of size [data_sample_length, n_dims]
        n_dims = sample_data.shape[-1]

        filtered_data = []
        for dim in range(n_dims):
            filtered_data.append(self._filter(sample_data[:, dim]))
        filtered_data = np.vstack(filtered_data)
        filtered_data = np.einsum('ij->ji', filtered_data)

        # set data for sample
        sample.data = filtered_data
        return sample

    def _filter(self, data: np.ndarray) -> np.ndarray:
        """
        Filters data with implementation from official Scipy package.
        For further descriptions see: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
        :param data: data for filtering,
        :type data: np.ndarray,
        :return: filtered data,
        :rtype: np.ndarray.
        """
        order_range = range(self.order + 1)
        half_window = (self.window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[self.derivative.value] * self.rate ** self.derivative.value * factorial(
            self.derivative.value)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = data[0] - np.abs(data[1:half_window + 1][::-1] - data[0])
        lastvals = data[-1] + np.abs(data[-half_window - 1:-1][::-1] - data[-1])
        data = np.concatenate((firstvals, data, lastvals))
        return np.convolve(m[::-1], data, mode='valid')

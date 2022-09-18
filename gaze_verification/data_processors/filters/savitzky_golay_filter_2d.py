import scipy
import itertools
import numpy as np
from tqdm import tqdm
from scipy import signal
from typeguard import typechecked
from typing import List, Union, Optional, Any, Tuple

from gaze_verification.data_objects.sample import Sample, Samples
from gaze_verification.data_processors.filters.filter_abstract import FilterAbstract
from gaze_verification.data_processors.filters.filtration_utils import DerivativeType


@typechecked
class SavitzkyGolayFilter2D(FilterAbstract):
    """
    Savitsky-Golay filter to smooth two dimensional data.
    1. for each point of the two dimensional matrix extract a sub-matrix,
        centered at that point and with a size equal to an odd number "window_size".
    2. for this sub-matrix compute a least-square fit of a polynomial surface,
        defined as p(x,y) = a0 + a1*x + a2*y + a3*x\^2 + a4*y\^2 + a5*x*y + ... .
        Note that x and y are equal to zero at the central point.
    3. replace the initial central point with the value computed with the fit.

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
                 derivative_type: Optional[Union[str, int, DerivativeType]] = DerivativeType.NONE,
                 keep_erroneous_samples: bool = True,
                 verbose: bool = True):
        super().__init__(verbose)
        self.window_size = window_size
        self.order = order
        # Number of terms in the polynomial expression
        self.n_terms = (self.order + 1) * (self.order + 2) / 2.0
        self.derivative_type = self._check_derivative_type(derivative_type)
        self.keep_erroneous_samples = keep_erroneous_samples
        self.check_selected_parameters()

    def check_selected_parameters(self):
        """
        Checks whether the selected window size and order pair is valid:
         - Window size is required to be a positive odd number, larger then 1,
         - Window size is required to be larger then selected order
            (1 point for each window side),
        """
        try:
            self.window_size = np.abs(int(self.window_size))
            self.order = np.abs(int(self.order))
        except ValueError as e:
            raise ValueError("Window size and order have to be of type 'int'")

        if self.window_size % 2 != 1 or self.window_size <= 1:
            raise TypeError("Window size must be a positive odd number larger then 1")

        if self.window_size ** 2 < self.n_terms:
            raise TypeError(f"Window size = {self.window_size} is too small for the polynomials order = {self.order}")

    def _check_derivative_type(self, derivative_type: Union[str, int, DerivativeType]) -> DerivativeType:
        """
        Check validness of derivative selection.
        """
        if not (isinstance(derivative_type, str) or isinstance(derivative_type, int) or isinstance(derivative_type,
                                                                                                   DerivativeType)):
            self._logger.error(f"Provided derivative should a type of `str`, `int` or `Derivative`, ",
                               f" provided parameter of type: {type(derivative_type)}")
            raise AttributeError(f"Provided derivative should a type of `str`, `int` or `Derivative`")

        if isinstance(derivative_type, str):
            if derivative_type not in DerivativeType.get_available_names():
                self._logger.error(f"Provided derivative should be one from available list: {DerivativeType.to_str()}",
                                   f" but was given: {derivative_type}")
                raise AttributeError(
                    f"Provided derivative should be one from available list: {DerivativeType.to_str()}")
            self._logger.info(f"Selected derivative: {derivative_type}")
            return getattr(DerivativeType, derivative_type)

        if isinstance(derivative_type, int):
            if derivative_type > len(DerivativeType):
                self._logger.error(f"Provided derivative type index is out of bounds for existing types. "
                                   f"It should be one from available list: {list(range(0, len(DerivativeType)))}"
                                   f" but was given: {derivative_type}")
                raise AttributeError(
                    f"Provided derivative type index is out of bounds for existing types. "
                    f"It should be one from available list: {list(range(1, len(DerivativeType) + 1))}"
                    f" but was given: {derivative_type}")
            self._logger.info(f"Selected derivative: {derivative_type}")
            return DerivativeType(derivative_type)
        return derivative_type

    def _check_data(self, data: Any) -> np.ndarray:
        """
        Check validness of derivative selection.
        """
        if not (isinstance(data, np.ndarray) or isinstance(data, list) or isinstance(data, tuple)):
            self._logger.error(f"Provided data should a type of `np.ndarray`, or array-like: `list` or `tuple`, ",
                               f" provided parameter is of type: {type(data)}")
            raise AttributeError(f"Provided data should a type of `np.ndarray`, or array-like: `list` or `tuple`")

        if isinstance(data, list) or isinstance(data, tuple):
            try:
                data = np.asarray(data)
            except Exception as e:
                self._logger.error(f"Provided data should an array-like: `list` or `tuple`,"
                                   " which is convertible to `np.ndarray` type.",
                                   f" Provided parameter is of type: {type(data)} and raised error:\n{e}")
                raise AttributeError(f"Provided data should be convertible to `np.ndarray`!\nIt raised error:\n{e}")

        n_dims = data.shape[-1]
        if n_dims % 2 != 0:
            self._logger.warning(f"Provided data has odd number of dimensions",
                                 " which is not suitable for 2D Savitsky-Golay filter."
                                 "Last dimension will be skipped.")
        return data

    def filter_dataset(self, samples: Samples, dim_subscript: List[Tuple[int, int]] = None, **kwargs) -> Samples:
        """
        Create a new dataset containing filtered Samples.

        :param samples: DataClass containing N filtered Samples
        :type samples: Instances

        :param dim_subscript: specifies the subscripts for pairwise aggregation of data dimensions
                            as a list of tuples of subscript labels.
                            Each tuple of indexes corresponds to desired pair of dimensions of data.
                            Example:
                            [(0, 1), (2, 3)] - groups the first and the second dimensions,
                                               and thirst with fourth dimensions of array in pairs.
                            [(0, 1), (0, 3)] - groups the first and the second dimensions,
                                               and also first with fourth dimensions of array in pairs.
                                               So indexes can be repeated in different dimensions pairs.
        :type dim_subscript: str,

        :return: Samples object containing N filtered Samples
        :rtype: Samples
        """
        filtered_samples = []
        for sample in tqdm(samples, total=len(samples), desc="Filtering dataset..."):
            try:
                filtered_samples.append(self.filter_sample(sample, dim_subscript=dim_subscript))
            except Exception as e:
                self._logger.error(f"Error occurred during dataset filtration: {e}"
                                   f"\nSkipping sample: {sample.guid}.")
                # keep sample even if it's data is not filtered
                if self.keep_erroneous_samples:
                    filtered_samples.append(sample)

        return Samples(filtered_samples)

    def _check_dim_subscript(self, dim_subscript: List[Tuple[int, int]], data_size: Tuple[int, int]):
        """
        Checks whether the selected subscript for pairwise aggregation of data dimensions is valid:
            - Minimum provided dimension index is > 0,
            - Maximum provided dimension index is less then existing dimensions in data.
        """
        dim_subscript_ = list(itertools.chain.from_iterable(dim_subscript))
        max_dim = max(dim_subscript_)
        num_less_zero = sum(list(map(lambda x: x < 0, dim_subscript_)))

        if max_dim > data_size[-1]:
            self._logger.error(f"Maximum provided dimension index ({max_dim}) is larger"
                               f" then number of existing dimensions in data ({data_size[-1]})")
            raise AttributeError(f"Maximum provided dimension index ({max_dim}) is larger"
                                 f" then number of existing dimensions in data ({data_size[-1]})")

        if num_less_zero > 0:
            self._logger.error(f"Some provided dimensions indexes are less then 0.")
            raise AttributeError(f"Some provided dimensions indexes are less then 0.")

    def filter_sample(self, sample: Sample, dim_subscript: List[Tuple[int, int]] = None, **kwargs) -> Sample:
        """
        Filter data sequences from Samples according to predefined logic.
        :param sample: Sample object containing information about one Sample
        :type sample: Sample

        :param dim_subscript: specifies the subscripts for pairwise aggregation of data dimensions
                            as a list of tuples of subscript labels.
                            Each tuple of indexes corresponds to desired pair of dimensions of data.
                            Example:
                            [(0, 1), (2, 3)] - groups the first and the second dimensions,
                                               and thirst with fourth dimensions of array in pairs.
                            [(0, 1), (0, 3)] - groups the first and the second dimensions,
                                               and also first with fourth dimensions of array in pairs.
                                               So indexes can be repeated in different dimensions pairs.
        :type dim_subscript: str,

        :return: Sample object with filtered data field,
        :rtype: Sample
        """
        sample_data = self._check_data(sample.data)  # initial of size [data_sample_length, n_dims]
        n_dims = sample_data.shape[-1]

        if dim_subscript is not None:
            self._check_dim_subscript(dim_subscript, sample_data.shape)
        else:
            # init with sequential dimensions pairs
            dim_subscript = [(i, i + 1) for i in range(sample_data.shape[-1] - 1)]

        filtered_data = []
        for dims in dim_subscript:
            filtered_data.append(self._filter(sample_data[:, dims]))
        filtered_data = np.hstack(filtered_data)

        # set data for sample
        sample.data = filtered_data
        return sample

    def _filter(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Filters data with implementation from official Scipy package.
        For further descriptions see: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
        :param data: data for filtering,
        :type data: np.ndarray,
        :return: filtered data,
        :rtype: np.ndarray.
        """
        half_size = self.window_size // 2

        # Exponents of the polynomial:
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
        # this line gives a list of two item tuple. Each tuple contains
        # the exponents of the k-th term. First element of tuple is for x second element for y.
        # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
        exps = [(k - n, n) for k in range(self.order + 1) for n in range(k + 1)]

        # Coordinates of points
        ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
        dx = np.repeat(ind, self.window_size)
        dy = np.tile(ind, [self.window_size, 1]).reshape(self.window_size ** 2, )

        # Build matrix of system of equation
        A = np.empty((self.window_size ** 2, len(exps)))
        for i, exp in enumerate(exps):
            A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

        # Pad input array with appropriate values at the four borders
        new_shape = data.shape[0] + 2 * half_size, data.shape[1] + 2 * half_size
        Z = np.zeros((new_shape))
        # Top band
        band = data[0, :]
        Z[:half_size, half_size:-half_size] = band - np.abs(
            np.flipud(data[1:half_size + 1, :]) - band)
        # Bottom band
        band = data[-1, :]
        Z[-half_size:, half_size:-half_size] = band + np.abs(
            np.flipud(data[-half_size - 1:-1, :]) - band)
        # Left band
        band = np.tile(data[:, 0].reshape(-1, 1), [1, half_size])
        Z[half_size:-half_size, :half_size] = band - np.abs(
            np.fliplr(data[:, 1:half_size + 1]) - band)
        # Right band
        band = np.tile(data[:, -1].reshape(-1, 1), [1, half_size])
        Z[half_size:-half_size,
        -half_size:] = band + np.abs(np.fliplr(data[:, -half_size - 1:-1]) - band)
        # Central band
        Z[half_size:-half_size, half_size:-half_size] = data

        # Top left corner
        band = data[0, 0]
        Z[:half_size, :half_size] = band - np.abs(
            np.flipud(np.fliplr(data[1:half_size + 1, 1:half_size + 1])) - band)
        # Bottom right corner
        band = data[-1, -1]
        Z[-half_size:, -half_size:] = band + np.abs(
            np.flipud(np.fliplr(data[-half_size - 1:-1, -half_size - 1:-1])) - band)

        # Top right corner
        band = Z[half_size, -half_size:]
        Z[:half_size, -half_size:] = band - np.abs(
            np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
        # Bottom left corner
        band = Z[-half_size:, half_size].reshape(-1, 1)
        Z[-half_size:, :half_size] = band - np.abs(
            np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

        # Solve system and convolve
        if self.derivative_type == DerivativeType.NONE:
            m = np.linalg.pinv(A)[0].reshape((self.window_size, -1))
            return signal.fftconvolve(Z, m, mode='valid')

        elif self.derivative_type == DerivativeType.COLUMN:
            c = np.linalg.pinv(A)[1].reshape(self.window_size, -1)
            return signal.fftconvolve(Z, -c, mode='valid')

        elif self.derivative_type == DerivativeType.ROW:
            r = np.linalg.pinv(A)[2].reshape((self.window_size, -1))
            return signal.fftconvolve(Z, -r, mode='valid')

        elif self.derivative_type == DerivativeType.BOTH:
            c = np.linalg.pinv(A)[1].reshape((self.window_size, -1))
            r = np.linalg.pinv(A)[2].reshape((self.window_size, -1))
            return signal.fftconvolve(
                Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

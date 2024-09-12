# Path and Char manipulation
import os
import sys
import pathlib
import re

# Tools
import copy
import itertools
import logging
from collections import deque

# Data processing
import numpy as np
import dask.array as da
import pandas as pd

# I/O
import io
import csv
import h5py as hdf
import ast

# Signal processing
import scipy as spy
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, filtfilt, savgol_filter, welch

# Orion
import Core as Orion
from SharedMethods import SharedMethods

# Message mode
import warnings
Verbose = Orion.DEFAULT_VERBOSE
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'

class Processor(SharedMethods):
    """
    Data processing
    """
    def __init__(self, base, quantities_of_interest = None,
                 NumericalScheme = None, gridFields = None):
        """
        Initialize the ExportDataToWrite subclass.

        :param base: The dataset containing the data to be exported.
        :type base: Dataset
        :param quantities_of_interest_dic: Dictionary specifying quantities of interest (default is None).
        :type quantities_of_interest_dic: dict, optional
        :param NumericalScheme: The numerical scheme used for export (default is None).
        :type NumericalScheme: str, optional
        :param gridFields: Fields related to the grid (default is None).
        :type gridFields: dict, optional

        .. note::

            - This method initializes the ExportDataToWrite subclass with provided parameters.
            - It sets attributes for the dataset (`base`), quantities of interest (`quantities_of_interest_dic`),
            numerical scheme (`NumericalScheme`), and grid fields (`gridFields`).
        """
        super().__init__()
        if quantities_of_interest is None:
            self.base = base
        elif isinstance(base, (list, dict)):
            self.print_text("error", "The reduction of variables currently supports only a single base input at a time.\n")
            sys.exit()
        else:
            self.quantities_of_interest = quantities_of_interest
            self.base = self.__reudce_variables(base, quantities_of_interest)

        self.NumericalScheme = NumericalScheme
        self.gridFields = gridFields

    def fusion(self):
        """
        Merge multiple data sources into a single pivot base.

        :param self: The instance of the class that this method belongs to.
        :type self: YourClassName

        :return: A merged data structure containing all data from the base sources.
        :rtype: PivotBase

        .. note::

            - This method initializes the merge with the first data source as the pivot base.
            - It iterates through the remaining data sources, updating the pivot base with zones, instants, and variables.
            - If a zone or instant is missing in the pivot base, it is added.
            - Duplicate variables within the same zone and instant are detected and renamed to avoid conflicts, with a warning message printed.
            - Ensure that variable names are unique to prevent unexpected behavior.
        """
        bases = deque(self.base)
        pivot_base = bases.popleft()

        for base in bases:
            for n, (zone, instant) in enumerate(base.items()):

                if zone not in pivot_base.keys():
                    if n == 0: pivot_base.add_zone(zone)
                    pivot_base[zone].add_instant(instant)
                    for var in base[zone][instant].keys():
                        pivot_base[zone][instant].add_variable(var,
                                                            base[zone][instant][var].data)
                else:
                    if instant not in pivot_base[zone].keys():
                        pivot_base[zone].add_instant(instant)
                        for var in base[zone][instant].keys():
                            pivot_base[zone][instant].add_variable(var,
                                                                base[zone][instant][var].data)
                    else:
                        for var in base[zone][instant].keys():
                            if var not in pivot_base[zone][instant].keys():
                                pivot_base[zone][instant].add_variable(var,
                                                                    base[zone][instant][var].data)
                            else:
                                self.print_text("warning", f"Warning duplicate variables detected : {var} renamed to {var}_DUP")
                                pivot_base[zone][instant].add_variable(f"{var}_DUP",
                                                                    base[zone][instant][var].data)
        return pivot_base

    def fft(self, decomposition_type="both", frequencies_band = (None, None), **kwargs):
        """
        Compute the Fast Fourier Transform (FFT) of the input signal for each variable in the dataset.

        :param dt: The time interval between samples in the input signal.
        :type dt: float
        :param decomposition_type: Specifies the type of decomposition to return. Options are:
            - "im/re": Returns the real and imaginary components.
            - "mod/phi": Returns the magnitude and phase components.
            - "both": Returns both real/imaginary and magnitude/phase components (default).
        :type decomposition_type: str, optional
        :param frequencies_band: A tuple or list defining the frequency band to apply for filtering.
            - If `None`, no frequency filtering is applied (default: (None, None)).
            - Values must be in the form (min_freq, max_freq). Frequencies outside this range are excluded.
        :type frequencies_band: tuple or list of float, optional
        :param kwargs: Additional arguments. This can include:
            - `invariant_variables`: A list of variable names to exclude from FFT computation.
            Default is set to Orion.DEFAULT_COORDINATE_NAMES + Orion.DEFAULT_TIME_NAME.

        :return: A dictionary-like object (fft_base) with the FFT results, varying by `decomposition_type`:
            - For "im/re":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **real**: The real part of the FFT result for each variable.
                - **imaginary**: The imaginary part of the FFT result for each variable.
            - For "mod/phi":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **magnitude**: The magnitude of the FFT result for each variable.
                - **phase**: The phase of the FFT result for each variable.
            - For "both":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **real**: The real part of the FFT result for each variable.
                - **imaginary**: The imaginary part of the FFT result for each variable.
                - **magnitude**: The magnitude of the FFT result for each variable.
                - **phase**: The phase of the FFT result for each variable.
        :rtype: dict-like (fft_base)

        :raises ValueError:
            - If `frequencies_band` is not a tuple or list, or if it does not contain exactly two elements.
            - If `decomposition_type` is not one of the accepted values ("im/re", "mod/phi", "both").
        :raises TypeError: If `frequencies_band` is not a tuple or list.

        .. note::
            - This function leverages Dask for parallel and out-of-core computation to handle large datasets.
            - Only positive frequencies are considered due to the symmetry of the FFT.
            - The `mask_band` method is used to filter out frequencies outside the specified `frequencies_band`.
            - `fft_base` contains FFT results for each variable that is not listed in `invariant_variables`.
            """
        invariant_variables = kwargs.get("invariant_variables",
                                         Orion.DEFAULT_TIME_NAME)

        if np.any(frequencies_band):
            if not isinstance(frequencies_band, (tuple, list)):
                print("Frequencies band type can be tuple or list.")
                raise ValueError

            if not isinstance(frequencies_band, (tuple, list)):
                raise TypeError("Frequencies band must be a tuple or list.")

            if len(frequencies_band) != 2:
                frequencies_band = [np.min(frequencies_band), np.max(frequencies_band)]

        if decomposition_type not in ["im/re", "mod/phi", "both", 'complex']:
            raise ValueError("Invalid decomposition_type. Choose from 'im/re', 'mod/phi', complex, or 'both'.")

        fft_base = Orion.Base()
        fft_base.add_zone(list(self.base.keys()))

        for zone, instant in self.base.items():
            fft_base[zone].add_instant(instant)

            time_name = Orion.DEFAULT_TIME_NAME[0]
            dt = (self.base[zone][instant][time_name][1] - self.base[zone][instant][time_name][0]).compute()

            for variable, value in self.base[zone][instant].items():

                if variable not in invariant_variables:
                    # Compute the FFT using Dask
                    fft_value = da.fft.fft(value, axis=0)
                    fft_freqs = da.fft.fftfreq(len(value), d=dt)
                    # Only take the positive frequencies (since FFT is symmetric)

                    if np.any(frequencies_band):
                        mask = self.__mask_band(fft_freqs[:len(fft_freqs)//2], frequencies_band)
                        fft_value = fft_value[:len(fft_value)//2][mask.compute()]
                        fft_freqs = fft_freqs[:len(fft_freqs)//2][mask.compute()]
                    else:
                        fft_value = fft_value[:len(fft_value)//2]
                        fft_freqs = fft_freqs[:len(fft_freqs)//2]

                    fft_base[zone][instant].add_variable(
                        Orion.DEFAULT_FREQUENCY_NAME[0], fft_freqs)

                    if decomposition_type in ["im/re", "both"]:
                        fft_base[zone][instant].add_variable(f"{variable}_real",
                                                             np.real(fft_value))
                        fft_base[zone][instant].add_variable(f"{variable}_img",
                                                             np.imag(fft_value))

                    if decomposition_type in ["mod/phi", "both"]:
                        fft_base[zone][instant].add_variable(f"{variable}_mag",
                                                             da.abs(fft_value))
                        fft_base[zone][instant].add_variable(f"{variable}_phase",
                                                             da.angle(fft_value))

                    if decomposition_type in ["complex"]:
                        fft_base[zone][instant].add_variable(f"{variable}_complex",
                                                             fft_value)

        return fft_base

    def psd(self, frequencies_band = (None, None), **kwargs):

        if np.any(frequencies_band):
            if not isinstance(frequencies_band, (tuple, list)):
                print("Frequencies band type can be tuple or list.")
                raise ValueError

            if not isinstance(frequencies_band, (tuple, list)):
                raise TypeError("Frequencies band must be a tuple or list.")

            if len(frequencies_band) != 2:
                frequencies_band = [np.min(frequencies_band), np.max(frequencies_band)]

        invariant_variables = kwargs.get("invariant_variables",
                                         Orion.DEFAULT_TIME_NAME)

        psd_base = Orion.Base()
        psd_base.add_zone(list(self.base.keys()))

        for zone, instant in self.base.items():
            psd_base[zone].add_instant(instant)

            time_name = Orion.DEFAULT_TIME_NAME[0]
            dt = (self.base[zone][instant][time_name][1] - self.base[zone][instant][time_name][0]).compute()

            for variable, value in self.base[zone][instant].items():

                if variable not in invariant_variables:

                    psd_freqs, psd_value = welch(value, fs=1/dt,
                                                 nperseg=len(value)//8)

                    if np.any(frequencies_band):
                        mask = self.__mask_band(psd_freqs, frequencies_band)
                        psd_freqs = psd_freqs[mask]
                        psd_value = psd_value[mask]

                    psd_base[zone][instant].add_variable(
                        Orion.DEFAULT_FREQUENCY_NAME[0], psd_freqs)
                    psd_base[zone][instant].add_variable(variable, psd_value)

        return psd_base

    def filter(self, **kwargs):
        """
        Apply a digital filter to the data using various filter types and configurations.

        :param filter_type: The type of filter to apply. Options include:
            - 'butterworth': Butterworth filter.
            - 'cheby1': Chebyshev Type I filter.
            - 'cheby2': Chebyshev Type II filter.
            - 'elliptic': Elliptic filter.
            - 'bessel': Bessel filter.
        Default is 'butterworth'.
        :type filter_type: str, optional

        :param cutoff: The cutoff frequency or frequencies for the filter:
            - For 'low' and 'high' filters, a single cutoff frequency.
            - For 'band' and 'stop' filters, a list or tuple of two cutoff frequencies.
        If `None`, no filtering is applied. Default is `None`.
        :type cutoff: float, list or tuple, optional

        :param sampling_rate: The rate at which samples were taken (samples per second). Default is 1000 Hz.
        :type sampling_rate: float, optional

        :param order: The order of the filter. Higher values mean a steeper roll-off. Default is 5.
        :type order: int, optional

        :param btype: The type of filter to design:
            - 'low': Low-pass filter. Allows frequencies below the cutoff to pass through.
            - 'high': High-pass filter. Allows frequencies above the cutoff to pass through.
            - 'band': Band-pass filter. Allows frequencies within the cutoff range to pass through.
            - 'stop': Band-stop (notch) filter. Blocks frequencies within the cutoff range.
        Default is 'low'.
        :type btype: str, optional

        :return: The modified data with the filter applied.
        :rtype: dict-like (self.base)

        :raises ValueError:
            - If `cutoff` is not appropriate for the specified `btype`.
            - If `cutoff` frequencies exceed the Nyquist frequency (half the sampling rate).
            - If an unsupported `filter_type` is specified.

        .. note::
            - The Nyquist frequency is half the sampling rate, and it is used to normalize the cutoff frequency.
            - The `filtfilt` function is used for zero-phase filtering, ensuring no phase distortion.
        """

        filter_type = kwargs.get('filter_type', 'butterworth')
        cutoff = kwargs.get('cutoff', None)
        sampling_rate = kwargs.get('sampling_rate', 1e3)
        order = kwargs.get('order', 5)
        btype = kwargs.get('btype', 'low')

        # Ensure cutoff is a list if btype is 'band' or 'stop'
        if btype in ['band', 'stop'] and not isinstance(cutoff, (list, tuple)):
            raise ValueError(f"For 'btype'='{btype}', 'cutoff' must be a list or tuple of two values.")
        elif btype not in ['band', 'stop'] and isinstance(cutoff, (list, tuple)):
            raise ValueError(f"For 'btype'='{btype}', 'cutoff' should not be a list or tuple.")

        nyquist_frequency = sampling_rate / 2.0
        if btype in ['band', 'stop']:
            normal_cutoff = [freq / nyquist_frequency for freq in cutoff]
        else:
            normal_cutoff = cutoff / nyquist_frequency

        if (btype in ['band', 'stop'] and (normal_cutoff[0] >= 1 or normal_cutoff[1] >= 1)) or (btype not in ['band', 'stop'] and normal_cutoff >= 1):
            raise ValueError("Cutoff frequency must be less than the Nyquist frequency.")

        if filter_type == 'butterworth':
            b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'cheby1':
            b, a = cheby1(order, 0.5, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'cheby2':
            b, a = cheby2(order, 20, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'elliptic':
            b, a = ellip(order, 0.5, 20, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'bessel':
            b, a = bessel(order, normal_cutoff, btype=btype, analog=False, norm='phase')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        for zone, instant in self.base.items():
            for variable_name, variable_obj in self.base[zone][instant].items():
                self.base[zone][instant].add_variable(variable_name,
                                                      filtfilt(b, a,
                                                               variable_obj)
                                                      )

        return self.base

    def reduce(self, factor = 2):
        reduce_base = Orion.Base()
        reduce_base.add_zone(list(self.base.keys()))
        for zone, instant in self.base.items():
            reduce_base[zone].add_instant(instant)
            for variable_name, variable_obj in list(self.base[zone][instant].items()):
                reduce_base[zone][instant].add_variable(variable_name,
                                                        variable_obj[::factor])

        return reduce_base

    def detrend(self, type = None, **kwargs):
        invariant_variables = kwargs.get("invariant_variables",
                                         Orion.DEFAULT_TIME_NAME +
                                         Orion.DEFAULT_FREQUENCY_NAME)
        if type is None:
            type  = 'constant'

        detrend_base = Orion.Base()
        detrend_base.add_zone(list(self.base.keys()))
        for zone, instant in self.base.items():
            detrend_base[zone].add_instant(instant)
            for variable_name, variable_obj in list(self.base[zone][instant].items()):
                if variable_name in invariant_variables:
                    detrend_base[zone][instant].add_variable(variable_name,
                                                             variable_obj)
                else:
                    detrend_base[zone][instant].add_variable(variable_name,
                                                             spy.signal.detrend(variable_obj, type = type))

        return detrend_base


    def smooth(self, window = None, order = None, **kwargs):

        if window is None:
            window = [5, 5, 5]

        if order is None:
            order = 1

        invariant_variables = kwargs.get("invariant_variables",
                                         Orion.DEFAULT_TIME_NAME +
                                         Orion.DEFAULT_FREQUENCY_NAME)

        smooth_base = Orion.Base()
        smooth_base.add_zone(list(self.base.keys()))
        for zone, instant in self.base.items():
            smooth_base[zone].add_instant(instant)
            for variable_name, variable_obj in list(self.base[zone][instant].items()):
                if variable_name not in invariant_variables:

                    if np.issubdtype(variable_obj, np.complexfloating):
                        self.print_text("error", f"Can not smooth complex number. Variable {variable_name} is complex.")
                        raise ValueError
                    else:
                        smoothed_variable = self.smoothing(variable_obj,
                                                        order,
                                                        *window)
                    smooth_base[zone][instant].add_variable(variable_name,
                                                            smoothed_variable)
                else:
                    smooth_base[zone][instant].add_variable(variable_name,
                                                            variable_obj)

        return smooth_base
    
    def linear_regression(self, independent_variable_name = None):
        
        if independent_variable_name is None:
            independent_variable_name = Orion.DEFAULT_TIME_NAME
        
        if not isinstance(independent_variable_name, list):
            independent_variable_name = [independent_variable_name]
        
        variable_list = self.variables_location(self.base)
        if not set(independent_variable_name).issubset(set(variable_list)):
            self.print_text("error", "Predictor variable is not in the base variable (it has to be present in all instants)")
            raise KeyError
        
        linear_base = Orion.Base()
        linear_base.add_zone(list(self.base.keys()))
        for zone, instant in self.base.items():
            linear_base[zone].add_instant(instant)
            independent_variable = self.base[zone][instant][independent_variable_name[0]].data
            for variable_name, variable_obj in list(self.base[zone][instant].items()):
                if variable_name not in independent_variable_name:
                    y_linear_regression, *attr_values = \
                        self.dask_linear_regression(variable_obj, 
                                                    independent_variable, 
                                                    stats = True)
                        
                    linear_base[zone][instant].add_variable(variable_name, 
                                                           y_linear_regression)
                        
                    attr_names = ["slope", 'intercept', 'residual_sq_sum', 
                                  'err_slope', 'err_intercept', 'residuals']
                    for attr_name, attr_value in zip(attr_names, attr_values):
                        linear_base[zone][instant][variable_name].set_attribute(attr_name, attr_value)
                else:
                    linear_base[zone][instant].add_variable(variable_name, variable_obj)
        
        return linear_base
                    
    @staticmethod
    def dask_linear_regression(y, x, stats = False):
        # Compute necessary statistics with Dask
        x_mean = x.mean()
        y_mean = y.mean()

        # Covariance of x and y
        cov_xy = ((x - x_mean) * (y - y_mean)).mean()
        # Variance of x
        var_x = ((x - x_mean) ** 2).mean()
        # Compute slope and intercept using Dask
        slope, intercept = da.compute(cov_xy / var_x, 
                                      y_mean - (cov_xy / var_x) * x_mean)

        # Calculate the extracted trend using the computed slope and intercept
        y_linear_regression = slope * x + intercept
        
        if stats:
            n = len(x)
            residuals = y - y_linear_regression
            # Residual Sum of Squares (RSS)
            rss = (residuals ** 2).sum()
            # Residual Standard Error (RSE)
            rse = da.sqrt(rss / (n - 2))
            # Variance of x (used in the denominator for standard errors)
            var_x = ((x - x_mean) ** 2).sum()
            # Standard error of slope
            se_slope = rse / da.sqrt(var_x)
            # Standard error of intercept
            se_intercept = rse * da.sqrt((1 / n) + (x_mean ** 2 / var_x))

            return y_linear_regression, slope, intercept, rse.compute(), \
                se_slope.compute(), se_intercept.compute(), residuals
        return y_linear_regression, slope, intercept

    
    @staticmethod
    def smoothing(input_signal, polyorder, first_window_size, middle_window_size, last_window_size):
        # Apply Savitzky-Golay filter to different parts of the signal
        # First part
        y_smooth_first = savgol_filter(input_signal[:first_window_size], first_window_size, polyorder)
        # Middle part
        y_smooth_middle = savgol_filter(input_signal[first_window_size:-last_window_size], middle_window_size, polyorder)
        # Last part
        y_smooth_last = savgol_filter(input_signal[-last_window_size:], last_window_size, polyorder)

        # Concatenate the results
        output_signal_smooth = np.concatenate([y_smooth_first, y_smooth_middle, y_smooth_last])

        return output_signal_smooth

    @staticmethod
    def compute_transfer_function(input_signal, output_signal, time, order = (1, 2),
                              method = 'scipy_minimize', initial_params = None,
                              freq_range = None):
        """
        Compute the transfer function between input and output signals.

        Parameters:
        - input_signal: array-like, the input force spy.signal.
        - output_signal: array-like, the output displacement spy.signal.
        - time: array-like, time vector corresponding to the signals.
        - order: tuple of (numerator_order, denominator_order) for the transfer function.
        - method: str, method to use for system identification ('scipy_minimize', 'arx', 'prony', 'pem').
        - initial_params: array-like, initial guess for coefficients. If None, defaults to ones.
        - freq_range: tuple of (min_freq, max_freq) for frequency response. If None, uses default range.

        Returns:
        - system_optimized: scipy.spy.signal.TransferFunction object, the optimized transfer function.
        - result_dict: dictionary containing Bode plot data and additional information.
        """
        input_signal, output_signal, time = map(Processor.dask_to_numpy, [input_signal, output_signal, time])

        # Ensure uniform time intervals
        if np.max(np.diff(time)) != np.min(np.diff(time)):
            time = np.linspace(np.min(time), np.max(time), len(time))
            # input_signal = np.interp(time, Processor.prepare_data(time), input_signal)
            # output_signal = np.interp(time, Processor.prepare_data(time), output_signal)

        num_order, den_order = order
        if initial_params is None:
            initial_params = np.ones(num_order + den_order + 1)

        params_optimized = Processor.fit_scipy_minimize(num_order, den_order,
                                                        initial_params,
                                                        input_signal,
                                                        output_signal,
                                                        time)

        num_optimized = params_optimized[:num_order + 1]
        den_optimized = np.concatenate(([1.], params_optimized[num_order + 1:]))
        system_optimized = spy.signal.TransferFunction(num_optimized, den_optimized)

        # Check stability
        is_stable = np.all(np.real(np.roots(den_optimized)) < 0)

        # Compute frequency response
        if freq_range:
            w = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]),
                            num=1000)
        else:
            w = None

        # Correctly compute Bode plot for continuous-time transfer function
        if system_optimized.dt is None:
            w, mag, phase = spy.signal.bode(system_optimized, w=w)
        else:
            w, mag, phase = spy.signal.dlti(system_optimized).bode(w=w)

        # Compute additional frequency response plots
        w, h = spy.signal.freqs(num_optimized, den_optimized, w)

        result_dict = {
            'frequencies': w,
            'magnitude': mag,
            'phase': phase,
            'is_stable': is_stable,
            'nyquist': h,
            'coherence': np.corrcoef(input_signal, output_signal)[0, 1],
            'params': params_optimized,
            'method': method,
            'order': order
        }

        return system_optimized, result_dict

    @staticmethod
    def dask_to_numpy(input):
        if isinstance(input, da.Array):
            return input.compute()
        return np.asarray(input)

    def __mask_band(self, input_signal, band):
        min_band, max_band = band

        # Ensure min_band and max_band are either None or numeric
        if min_band is not None and not isinstance(min_band, (int, float)):
            raise ValueError("min_band must be a numeric value or None.")

        if max_band is not None and not isinstance(max_band, (int, float)):
            raise ValueError("max_band must be a numeric value or None.")

        # Handle various cases for min_band and max_band
        if min_band is not None and max_band is not None:
            if min_band >= max_band:
                raise ValueError("min_band cannot be greater than max_band.")
            mask = (input_signal >= min_band) & (input_signal <= max_band)

        elif min_band is not None and max_band is None:
            mask = input_signal >= min_band

        elif min_band is None and max_band is not None:
            mask = input_signal <= max_band

        else:
            raise ValueError("At least one of min_band or max_band must be provided.")

        return mask

    def __numpy_to_dask(self, input, chunk_size="auto"):
        if isinstance(input, np.ndarray):
            return da.from_array(input, chunks=chunk_size)
        return input

    def __rechunk(self, input, chunk_size="auto"):
        if isinstance(input, np.ndarray):
            return self.__numpy_to_dask(input, chunk_size)
        else:
            return input.rechunk((chunk_size))


    @staticmethod
    def fit_scipy_minimize(num_order, den_order, initial_params, input_signal,
                           output_signal, time):
        """ Fit the transfer function using scipy's minimize. """
        def fit_transfer_function(params):
            num = params[:num_order + 1]
            den = np.concatenate(([1.], params[num_order + 1:]))
            system = spy.signal.TransferFunction(num, den)
            _, yout, _ = spy.signal.lsim(system, U=input_signal, T=time)
            return np.sum((yout - output_signal) ** 2)

        result = spy.optimize.minimize(fit_transfer_function, initial_params,
                          method='Nelder-Mead')
        return result.x


if __name__ == "__main__":
    # ============================ Case root path ==============================
    # Case Root Path
    path = r"T:\CSprojects\CS-projects\CS13124  CV-RTD valve ext\BHG_Data"
    case_path = SharedMethods().path_manage(path)

    # Inputs Path
    input_common_path = os.path.join(case_path, "EWR13124-876", "G60_Perfo")
    input_name_patterns = [
        "G60-<instant>.MAT"
        ]
    # ============================== Read inputs ===============================
    file_dic = Reader(input_common_path, input_name_patterns).read_mat()


    file_path = os.path.join(input_common_path, list(file_dic.keys())[0])
    file = spy.io.loadmat(file_path)


    # ============================= Data manipulation ==========================
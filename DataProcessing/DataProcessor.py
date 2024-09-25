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

        :param time_step: The time interval between samples in the input signal.
            - This is computed from the variable time in the instant or can be passed as an attribute of the instant, otherwiese it is set to 1.
        :type time_step: float
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
        time_name = Orion.DEFAULT_TIME_NAME[0]
        time_step_name = Orion.DEFAULT_TIMESTEP_NAME[0]

        if np.any(frequencies_band):
            if not isinstance(frequencies_band, (tuple, list)):
                print("Frequencies band type can be tuple or list.")
                raise ValueError

            if not isinstance(frequencies_band, (tuple, list)):
                raise TypeError("Frequencies band must be a tuple or list.")

            if len(frequencies_band) != 2:
                frequencies_band = [np.min(frequencies_band), np.max(frequencies_band)]

        if decomposition_type not in ["im/re", "mod/phi", "both", 'complex']:
            raise ValueError("Invalid decomposition_type. Choose from 'im/re'," 
                             "'mod/phi', complex, or 'both'.")

        fft_base = Orion.Base()
        fft_base.add_zone(list(self.base.keys()))

        for zone, instant in self.base.items():
            fft_base[zone].add_instant(instant)

            # Compute the sampling frequency for each variable fs = 1/dt
            if time_name in self.base[zone][instant].keys():
                time_step = (self.base[zone][instant][time_name].data.ravel()[1] - \
                    self.base[zone][instant][time_name].data.ravel()[0]).compute()
            else:
                time_step = self.base[zone][instant].get_attribute(time_step_name)
                if time_step is None:
                    self.print_text("warning", f"Neither TimeValue nor TimeStep were found in the variables instant: {instant} or in the instant attribute, therefore the time step is assumed to be 1/Number of points.")
                    
            for variable, value in self.base[zone][instant].items():

                if variable not in invariant_variables and value is not None:

                    if time_step is None:
                        time_step = len(value)
                        
                    # Compute the FFT using Dask
                    fft_value = da.fft.fft(value, axis=0)
                    fft_freqs = da.fft.fftfreq(len(value), d=time_step)
                    
                    # Only take the positive frequencies (since FFT is symmetric)
                    if np.any(frequencies_band):
                        mask = self.__mask_band(fft_freqs[:len(fft_freqs)//2], 
                                                frequencies_band)
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
        time_name = Orion.DEFAULT_TIME_NAME[0]
        time_step_name = Orion.DEFAULT_TIMESTEP_NAME[0]

        psd_base = Orion.Base()
        psd_base.add_zone(list(self.base.keys()))

        for zone, instant in self.base.items():
            psd_base[zone].add_instant(instant)

           # Compute the sampling frequency for each variable fs = 1/dt
            if time_name in self.base[zone][instant].keys():
                time_step = (self.base[zone][instant][time_name].data.ravel()[1] - \
                    self.base[zone][instant][time_name].data.ravel()[0]).compute()
            else:
                time_step = self.base[zone][instant].get_attribute(time_step_name)
                if time_step is None:
                    self.print_text("warning", f"Neither TimeValue nor TimeStep were found in the variables instant: {instant} or in the instant attribute, therefore the time step is assumed to be 1/Number of points.")

            for variable, value in self.base[zone][instant].items():

                if variable not in invariant_variables and value is not None:

                    if time_step is None:
                        time_step = 1/len(value)
                        
                    psd_freqs, psd_value = welch(value, fs=1/time_step,
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
        invariant_variables = kwargs.get("invariant_variables",
                                         Orion.DEFAULT_TIME_NAME +
                                         Orion.DEFAULT_FREQUENCY_NAME)

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

        filter_base = copy.deepcopy(self.base)
        for zone, instant in self.base.items():
            for variable_name, variable_obj in self.base[zone][instant].items():
                if variable_name not in invariant_variables:
                    filter_base[zone][instant].add_variable(variable_name,
                                                        filtfilt(b, a,
                                                                variable_obj)
                                                        )
        return filter_base

    def reduce(self, factor = 2):
        """
        Reduce the size of each variable in the base by a given factor.

        Parameters
        ----------
        factor : int, optional
            The reduction factor for the variables. Default is 2, meaning every other
            value will be taken.

        Returns
        -------
        reduce_base : Orion.Base
            A new base object with the reduced variables.

        Example
        -------
        >>> reduced_base = base.reduce(factor=2)
        """
        reduce_base = Orion.Base()
        reduce_base.add_zone(list(self.base.keys()))
        for zone, instant in self.base.items():
            reduce_base[zone].add_instant(instant)
            for variable_name, variable_obj in list(self.base[zone][instant].items()):
                reduce_base[zone][instant].add_variable(variable_name,
                                                        variable_obj[::factor])

        return reduce_base

    def detrend(self, type = None, **kwargs):
        """
        Detrend the variables in the base object, removing a specified trend type from the data.

        Parameters
        ----------
        type : str, optional
            Type of detrending to apply. Options include 'constant' or 'linear'. Default is 'constant'.
        **kwargs : dict, optional
            Additional arguments, including:
            - invariant_variables : list or str, optional
                Variables that should not be detrended. Defaults to time and frequency variables.

        Returns
        -------
        detrend_base : Orion.Base
            A new base object with the detrended variables.

        Example
        -------
        >>> detrended_base = base.detrend(type='linear')
        """
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
        """
        Apply a smoothing filter to the variables in the base object.

        Parameters
        ----------
        window : list of int, optional
            The size of the smoothing window along each axis. Default is [5, 5, 5].
        order : int, optional
            The order of the smoothing filter. Default is 1.
        **kwargs : dict, optional
            Additional arguments, including:
            - invariant_variables : list or str, optional
                Variables that should not be smoothed. Defaults to time and frequency variables.

        Returns
        -------
        smooth_base : Orion.Base
            A new base object with the smoothed variables.

        Raises
        ------
        ValueError
            If an attempt is made to smooth complex number data.

        Example
        -------
        >>> smoothed_base = base.smooth(window=[3, 3, 3], order=2)
        """
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

    def linear_regression(self, independent_variable_name = None, method = 'dask'):
        """
        Perform linear regression on the variables in the base object using the specified independent variable.

        Parameters
        ----------
        independent_variable_name : str or list of str, optional
            The name of the independent variable (predictor) to be used for the regression. 
            If not provided, defaults to the default time variable in Orion.
        method : str, optional
            The method to be used for performing the linear regression. Defaults to 'dask'.
            Other methods are : "robust" for RANSACRegressor.

        Returns
        -------
        linear_base : Orion.Base
            A new base object with the regression results for the dependent variables.

        Raises
        ------
        KeyError
            If the specified independent variable is not present in all instants of the base object.

        Notes
        -----
        - For each dependent variable (excluding the independent variable), a linear regression is performed.
        - Regression attributes such as slope, intercept, residual sum of squares, and errors are stored as variable attributes.

        Example
        -------
        >>> linear_base = base.linear_regression(independent_variable_name='Time')
        """
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
            
            if independent_variable_name[0] not in self.base[zone][instant].keys():
                continue
            
            linear_base[zone].add_instant(instant)
            independent_variable = self.base[zone][instant][independent_variable_name[0]].data
            for variable_name, variable_obj in list(self.base[zone][instant].items()):
                if variable_name not in independent_variable_name:
                    if method == "robust":
                        y_linear_regression, *attr_values = \
                            self.robust_linear_regression(variable_obj,
                                                        independent_variable,
                                                        stats = True)
                        print("toto")    
                    else:
                        print("lala")    
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
        """
        Perform linear regression on arrays using Dask for parallel computation, optionally returning statistical metrics.

        Parameters
        ----------
        y : dask.array.Array
            Dependent variable (response) array.
        x : dask.array.Array
            Independent variable (predictor) array.
        stats : bool, optional
            If True, return additional statistical metrics such as residuals and standard errors. Default is False.

        Returns
        -------
        y_linear_regression : dask.array.Array
            The predicted values from the linear regression.
        slope : float
            The slope of the linear regression line.
        intercept : float
            The intercept of the linear regression line.
        
        If stats=True, additionally returns:
        rse : float
            Residual Standard Error.
        se_slope : float
            Standard error of the slope.
        se_intercept : float
            Standard error of the intercept.
        residuals : dask.array.Array
            The residuals (differences between the actual and predicted values).

        Example
        -------
        >>> y_pred, slope, intercept = Processor.dask_linear_regression(y, x)
        >>> y_pred, slope, intercept, rse, se_slope, se_intercept, residuals = Processor.dask_linear_regression(y, x, stats=True)
        """
        # Compute necessary statistics with Dask
        x_mean = x.mean()
        y_mean = y.mean()

        # Covariance of x and y
        cov_xy = ((x - x_mean) * (y - y_mean)).mean()
        # Variance of x
        var_x = ((x - x_mean) ** 2).mean()
        if var_x == 0:
                raise ValueError("Variance of x is zero, cannot compute linear regression.")
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
            # Standard error of slope
            se_slope = rse / da.sqrt(var_x)
            # Standard error of intercept
            se_intercept = rse * da.sqrt((1 / n) + (x_mean ** 2 / var_x))

            return y_linear_regression, slope, intercept, rse.compute(), \
                se_slope.compute(), se_intercept.compute(), residuals
        return y_linear_regression, slope, intercept
    
    @staticmethod
    def robust_linear_regression(y, x, stats=False):
        """
        Perform robust linear regression on arrays using the RANSAC algorithm, optionally returning statistical metrics.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (response) array.
        x : np.ndarray
            Independent variable (predictor) array.
        stats : bool, optional
            If True, return additional statistical metrics such as residuals and standard errors. Default is False.

        Returns
        -------
        y_linear_regression : np.ndarray
            The predicted values from the robust linear regression.
        slope : float
            The slope of the robust linear regression line.
        intercept : float
            The intercept of the robust linear regression line.

        If stats=True, additionally returns:
        rse : float
            Residual Standard Error.
        se_slope : float
            Standard error of the slope.
        se_intercept : float
            Standard error of the intercept.
        residuals : np.ndarray
            The residuals (differences between the actual and predicted values).

        Example
        -------
        >>> y_pred, slope, intercept = Processor.robust_linear_regression(y, x)
        >>> y_pred, slope, intercept, rse, se_slope, se_intercept, residuals = Processor.robust_linear_regression(y, x, stats=True)
        """
        from sklearn.linear_model import RANSACRegressor
        # Fit the RANSAC model
        model = RANSACRegressor()
        model.fit(x.reshape(-1, 1), y)
        
        # Get slope and intercept
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        # Calculate the predicted trend (y_linear_regression)
        y_linear_regression = slope * x + intercept
        
        if stats:
            n = len(x)
            # Calculate residuals
            residuals = y - y_linear_regression
            # Residual Sum of Squares (RSS)
            rss = np.sum(residuals ** 2)
            # Residual Standard Error (RSE)
            rse = np.sqrt(rss / (n - 2))
            # Variance of x
            x_mean = np.mean(x)
            var_x = np.sum((x - x_mean) ** 2)
            # Standard error of slope
            se_slope = rse / np.sqrt(var_x)
            # Standard error of intercept
            se_intercept = rse * np.sqrt((1 / n) + (x_mean ** 2 / var_x))
            
            return y_linear_regression, slope, intercept, rse.compute(), \
                se_slope.compute(), se_intercept.compute(), residuals
        
        return y_linear_regression, slope, intercept

    
    @staticmethod
    def smoothing(input_signal, polyorder, first_window_size, middle_window_size, last_window_size):
        """
        Apply Savitzky-Golay filter to smooth different parts of the signal with varying window sizes.

        Parameters
        ----------
        input_signal : numpy.array
            The input signal to be smoothed.
        polyorder : int
            The order of the polynomial to use in the Savitzky-Golay filter.
        first_window_size : int
            The window size for smoothing the first part of the signal.
        middle_window_size : int
            The window size for smoothing the middle part of the signal.
        last_window_size : int
            The window size for smoothing the last part of the signal.

        Returns
        -------
        output_signal_smooth : numpy.array
            The smoothed signal obtained by applying the Savitzky-Golay filter to different parts of the input signal.

        Example
        -------
        >>> smoothed_signal = Processor.smoothing(input_signal, polyorder=3, first_window_size=5, middle_window_size=11, last_window_size=5)
        """
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
        Compute the transfer function between input and output signals using optimization.

        Parameters
        ----------
        input_signal : numpy.array or dask.array
            The input signal to the system.
        output_signal : numpy.array or dask.array
            The output signal from the system.
        time : numpy.array or dask.array
            Time values corresponding to the signals.
        order : tuple of int, optional
            The order of the numerator and denominator of the transfer function. Default is (1, 2).
        method : str, optional
            The optimization method to use. Default is 'scipy_minimize'.
        initial_params : array-like, optional
            Initial guess for the transfer function parameters. If None, defaults to ones. Default is None.
        freq_range : tuple of float, optional
            The frequency range for computing the frequency response. If None, uses a default range. Default is None.

        Returns
        -------
        system_optimized : scipy.signal.TransferFunction
            The optimized transfer function system.
        result_dict : dict
            A dictionary containing the computed results including:
            - 'frequencies': Frequency values.
            - 'magnitude': Magnitude of the frequency response.
            - 'phase': Phase of the frequency response.
            - 'is_stable': Boolean indicating whether the system is stable.
            - 'nyquist': Frequency response for the Nyquist plot.
            - 'coherence': Coherence between input and output signals.
            - 'params': Optimized parameters of the transfer function.
            - 'method': The optimization method used.
            - 'order': The order of the numerator and denominator.

        Example
        -------
        >>> system_optimized, result = Processor.compute_transfer_function(input_signal, output_signal, time, order=(1, 2))

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
        """
        Convert a Dask array or any array-like input to a NumPy array.

        Parameters
        ----------
        input : dask.array.Array or array-like
            The input data to convert. If it's a Dask array, it will be computed and converted to a NumPy array. 
            If it's already a NumPy array or similar, it will be converted to a NumPy array without additional computation.

        Returns
        -------
        numpy.ndarray
            The converted NumPy array.

        Example
        -------
        >>> import dask.array as da
        >>> dask_array = da.from_array(np.array([1, 2, 3]), chunks=1)
        >>> numpy_array = Processor.dask_to_numpy(dask_array)
        >>> print(numpy_array)
        [1 2 3]

        """
        if isinstance(input, da.Array):
            return input.compute()
        return np.asarray(input)

    @staticmethod
    def fit_scipy_minimize(num_order, den_order, initial_params, input_signal,
                           output_signal, time):
        """
        Fit a transfer function to data using SciPy's minimize function.

        Parameters
        ----------
        num_order : int
            The order of the numerator polynomial of the transfer function.
            
        den_order : int
            The order of the denominator polynomial of the transfer function.
            
        initial_params : array-like
            Initial guess for the parameters of the transfer function. This should be a 1D array where
            the first `num_order + 1` elements are the numerator coefficients and the remaining elements
            are the denominator coefficients (excluding the leading 1 which is implicitly part of the denominator).
            
        input_signal : array-like
            The input signal data used for fitting the transfer function.
            
        output_signal : array-like
            The output signal data used for fitting the transfer function. This is compared to the response of
            the transfer function to the `input_signal`.
            
        time : array-like
            The time vector corresponding to the `input_signal` and `output_signal`.

        Returns
        -------
        numpy.ndarray
            The optimized parameters of the transfer function. This array contains the numerator coefficients
            followed by the denominator coefficients (excluding the leading 1).

        Example
        -------
        >>> num_order = 2
        >>> den_order = 2
        >>> initial_params = np.array([1, 1, 1, 1])
        >>> input_signal = np.array([0, 1, 2, 3])
        >>> output_signal = np.array([0, 0.9, 1.8, 2.7])
        >>> time = np.array([0, 1, 2, 3])
        >>> optimized_params = Processor.fit_scipy_minimize(num_order, den_order, initial_params, input_signal, output_signal, time)
        >>> print(optimized_params)
        [0.9 0.8 1.1 0.7]
        """
        
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
    
    def __mask_band(self, input_signal, band):
        """
        Create a boolean mask for the input signal based on the specified frequency band.

        Parameters
        ----------
        input_signal : array-like
            The signal data to be masked. This could be a NumPy array or a similar array-like structure.
            
        band : tuple
            A tuple defining the frequency band for masking. The tuple should contain two elements:
            - min_band : float or None
                The minimum value of the frequency band. If None, no lower bound is applied.
            - max_band : float or None
                The maximum value of the frequency band. If None, no upper bound is applied.

        Returns
        -------
        numpy.ndarray
            A boolean array where True indicates that the corresponding values in `input_signal` fall within the specified band.

        Raises
        ------
        ValueError
            If `min_band` is greater than or equal to `max_band`, or if neither `min_band` nor `max_band` is provided, or if either `min_band` or `max_band` is not a numeric value (when not None).

        Example
        -------
        >>> input_signal = np.array([1, 2, 3, 4, 5])
        >>> band = (2, 4)
        >>> mask = self.__mask_band(input_signal, band)
        >>> print(mask)
        [False  True  True  True False]

        """
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
        """
        Convert a NumPy array to a Dask array with specified chunk size.

        Parameters
        ----------
        input : numpy.ndarray or other
            The input data to convert. If it is a NumPy array, it will be converted to a Dask array.
            
        chunk_size : int, tuple, or "auto", optional
            The chunk size to use for the Dask array. If "auto", the default chunk size will be used.
            If an integer, it will be used for all dimensions. If a tuple, it specifies chunk sizes for
            each dimension. Default is "auto".

        Returns
        -------
        dask.array.Array or other
            The converted Dask array if the input was a NumPy array. If the input was not a NumPy array,
            it is returned unchanged.

        Example
        -------
        >>> import numpy as np
        >>> import dask.array as da
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> dask_arr = __numpy_to_dask(arr, chunk_size=2)
        >>> type(dask_arr)
        <class 'dask.array.core.Array'>
        
        """
        if isinstance(input, np.ndarray):
            return da.from_array(input, chunks=chunk_size)
        return input

    def __rechunk(self, input, chunk_size="auto"):
        """
        Rechunk an array or convert a NumPy array to a Dask array with specified chunk size.

        Parameters
        ----------
        input : numpy.ndarray or dask.array.Array
            The input data to rechunk. If it is a NumPy array, it will be converted to a Dask array
            and rechunked. If it is already a Dask array, its chunks will be adjusted.
            
        chunk_size : int, tuple, or "auto", optional
            The chunk size to use for the Dask array. If "auto", the default chunk size will be used.
            If an integer, it will be used for all dimensions. If a tuple, it specifies chunk sizes for
            each dimension. Default is "auto".

        Returns
        -------
        dask.array.Array or other
            The rechunked Dask array if the input was a NumPy array. If the input was already a Dask array,
            it returns the rechunked array.

        Example
        -------
        >>> import numpy as np
        >>> import dask.array as da
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> rechunked_arr = __rechunk(arr, chunk_size=2)
        >>> type(rechunked_arr)
        <class 'dask.array.core.Array'>
        
        >>> dask_arr = da.from_array([1, 2, 3, 4, 5], chunks=2)
        >>> rechunked_dask_arr = __rechunk(dask_arr, chunk_size=3)
        >>> rechunked_dask_arr.chunks
        ((3,),)
        """
        if isinstance(input, np.ndarray):
            return self.__numpy_to_dask(input, chunk_size)
        else:
            return input.rechunk((chunk_size))
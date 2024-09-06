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
import scipy as spy
import pandas as pd

# I/O
import io
import csv
import h5py as hdf
import ast

# Signal processing
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, filtfilt
from scipy import signal
from scipy.optimize import minimize

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

    @staticmethod
    def compute_fft(input_signal, dt, decomposition_type="both", frequencies_band = (None, None)):
        """
        Compute the Fast Fourier Transform (FFT) of the input signal.

        :param input_signal: The signal to transform, which can be a Dask array or a NumPy array.
        :type input_signal: da.Array or np.ndarray
        :param dt: The time interval between samples in the input signal.
        :type dt: float
        :param decomposition_type: The type of decomposition to return. Options are "im/re" for real and imaginary parts, 
                                    "mod/phi" for magnitude and phase, or "both" for all components. Default is "both".
        :type decomposition_type: str, optional
        :param frequencies_band: A tuple or list specifying the frequency band to select from the FFT result. 
                                  If None, no band filtering is applied. Default is (None, None).
        :type frequencies_band: tuple or list of float, optional

        :return: A dictionary containing the FFT result, which varies based on the `decomposition_type`:
            
            - If `decomposition_type` is "im/re":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **real**: The real part of the FFT result.
                - **imaginary**: The imaginary part of the FFT result.
            - If `decomposition_type` is "mod/phi":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **magnitude**: The magnitude of the FFT result.
                - **phase**: The phase of the FFT result.
            - If `decomposition_type` is "both":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **real**: The real part of the FFT result.
                - **imaginary**: The imaginary part of the FFT result.
                - **magnitude**: The magnitude of the FFT result.
                - **phase**: The phase of the FFT result.
        :rtype: dict

        :raises ValueError: If `frequencies_band` is not a tuple or list, or if it does not contain exactly two elements.
        :raises TypeError: If `frequencies_band` is not a tuple or list.
        :raises ValueError: If `decomposition_type` is not one of the accepted values ("im/re", "mod/phi", "both").

        .. note::

            - The FFT result is computed using Dask, allowing for parallel and out-of-core computations.
            - Only positive frequencies are considered due to the symmetry of the FFT.
            - The `mask_band` method of the `Processor` class is used to filter the frequency band if specified.
        """
        
        # Check if the input signal is a Dask array, if not, convert it
        if not isinstance(input_signal, da.Array):
            input_signal = da.from_array(input_signal, chunks=(len(input_signal),))  # Convert to Dask array with a single chunk

        # Check if the Dask array has a single chunk along the axis
        if input_signal.chunks[0][0] != len(input_signal):
            # Rechunk to a single chunk along the axis
            input_signal = input_signal.rechunk((len(input_signal),))

        # Compute the FFT using Dask
        fft_result_dask = da.fft.fft(input_signal, axis=0)

        # Compute the frequencies using Dask
        fft_freqs_dask = da.fft.fftfreq(len(input_signal), d=dt)

        # Only take the positive frequencies (since FFT is symmetric)
        positive_freqs_dask = fft_freqs_dask[:len(fft_freqs_dask)//2]
        positive_fft_dask = fft_result_dask[:len(fft_result_dask)//2]

        if np.any(frequencies_band):
            if not isinstance(frequencies_band, (tuple, list)):
                print("Frequencies band type can be tuple or list.")
                raise ValueError
            
            if not isinstance(frequencies_band, (tuple, list)):
                raise TypeError("Frequencies band must be a tuple or list.")
            
            if len(frequencies_band) != 2:
                raise ValueError("Frequencies band must contain exactly two elements (min_band, max_band).")
            
            mask = Processor.mask_band(positive_freqs_dask, frequencies_band)
            
            positive_freqs_dask = positive_freqs_dask[mask.compute()]
            positive_fft_dask = positive_fft_dask[mask.compute()]
        
        if decomposition_type == "im/re":
            # Return real and imaginary parts
            output_signal_dict = {
                "frequencies": positive_freqs_dask,
                "real": np.real(positive_fft_dask),
                "imaginary": np.imag(positive_fft_dask)
            }

        elif decomposition_type == "mod/phi":
            # Return magnitude and phase
            magnitude_dask = da.abs(positive_fft_dask)
            phase_dask = da.angle(positive_fft_dask)
            output_signal_dict = {
                "frequencies": positive_freqs_dask,
                "magnitude": magnitude_dask,
                "phase": phase_dask
            }

        elif decomposition_type == "both":
            # Return frequencies, real, imaginary, magnitude, and phase
            magnitude_dask = da.abs(positive_fft_dask)
            phase_dask = da.angle(positive_fft_dask)
            output_signal_dict = {
                "frequencies": positive_freqs_dask,
                "real": np.real(positive_fft_dask),
                "imaginary": np.imag(positive_fft_dask),
                "magnitude": magnitude_dask,
                "phase": phase_dask
            }

        else:
            raise ValueError("Invalid decomposition_type. Choose from 'im/re', 'mod/phi', or 'both'.")

        return output_signal_dict

    @staticmethod
    def filter(input_signal, config=None):
        """
        Apply one or more filters to the input signal based on the provided configuration.

        :param input_signal: The signal to be filtered.
        :type input_signal: array-like

        :param config: A dictionary containing the filter configuration. Possible keys include:
        
            - **filter_type**: The type of filter to apply. Can be a single type or a list of types. Options include 'butterworth', 'cheby1', 'cheby2', 'elliptic', 'bessel'. Default is 'butterworth'.
            - **cutoff**: The cutoff frequency or frequencies for the filter. For 'band' or 'stop' types, this should be a list or tuple of two values. For other types, it should be a single value. Default is 10.
            - **sampling_rate**: The sampling rate of the input signal. Default is `len(input_signal) / 10.0`.
            - **order**: The order of the filter. Default is 5.
            - **btype**: The type of filter response. Options are 'low', 'high', 'band', or 'stop'. Default is 'low'.
        :type config: dict, optional

        :return: The filtered signal or a dictionary of filtered signals if multiple filter types are used.
        :rtype: array-like or dict

        :raises ValueError: If the configuration parameters are invalid, such as incorrect cutoff frequencies or unsupported filter types.

        .. note::

            - The function supports various filter types including Butterworth, Chebyshev (both types), Elliptic, and Bessel.
            - For band or stop filters, ensure that `cutoff` is a list or tuple with two values.
            - The Nyquist frequency is used to normalize cutoff frequencies. Ensure that cutoff frequencies are less than the Nyquist frequency.
            - If only one filter type is provided, the function returns the filtered signal directly. If multiple filter types are provided, it returns a dictionary where keys are filter types and values are the corresponding filtered signals.
        """


        if config is None:
            config = {}

        filter_types = config.get('filter_type', 'butterworth')
        cutoff = config.get('cutoff', 10)
        sampling_rate = config.get('sampling_rate', len(input_signal) / 10.0)
        order = config.get('order', 5)
        btype = config.get('btype', 'low')

        # Ensure filter_types is a list for consistent processing
        if isinstance(filter_types, str):
            filter_types = [filter_types]

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

        # Dictionary to hold the output signals
        output_signals = {}

        # Iterate over each filter type
        for filter_type in filter_types:
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

            # Apply the filter and store the result
            output_signals[filter_type] = filtfilt(b, a, input_signal)

        # Return a single output if only one filter type was provided, otherwise return the dictionary
        if len(output_signals) == 1:
            return list(output_signals.values())[0]
        else:
            return output_signals

    @staticmethod
    def compute_transfer_function(input_signal, output_signal, time, order = (1, 2),
                              method = 'scipy_minimize', initial_params = None,
                              freq_range = None):
        """
        Compute the transfer function between input and output signals.

        Parameters:
        - input_signal: array-like, the input force signal.
        - output_signal: array-like, the output displacement signal.
        - time: array-like, time vector corresponding to the signals.
        - order: tuple of (numerator_order, denominator_order) for the transfer function.
        - method: str, method to use for system identification ('scipy_minimize', 'arx', 'prony', 'pem').
        - initial_params: array-like, initial guess for coefficients. If None, defaults to ones.
        - freq_range: tuple of (min_freq, max_freq) for frequency response. If None, uses default range.

        Returns:
        - system_optimized: scipy.signal.TransferFunction object, the optimized transfer function.
        - result_dict: dictionary containing Bode plot data and additional information.
        """
        input_signal, output_signal, time = map(Processor.prepare_data, [input_signal, output_signal, time])

        # Ensure uniform time intervals
        if np.max(np.diff(time)) != np.min(np.diff(time)):
            time = np.linspace(np.min(time), np.max(time), len(time))
            # input_signal = np.interp(time, Processor.prepare_data(time), input_signal)
            # output_signal = np.interp(time, Processor.prepare_data(time), output_signal)

        num_order, den_order = order
        if initial_params is None:
            initial_params = np.ones(num_order + den_order + 1)

        if method == 'scipy_minimize':
            params_optimized = Processor.fit_scipy_minimize(num_order, den_order, 
                                                            initial_params, 
                                                            input_signal, 
                                                            output_signal, 
                                                            time)
        elif method == 'arx':
            params_optimized = Processor.fit_arx(num_order, den_order, 
                                                 input_signal, output_signal)
        elif method == 'prony':
            params_optimized = Processor.fit_prony(num_order, den_order, 
                                                   output_signal)
        elif method == 'pem':
            params_optimized = Processor.fit_pem(num_order, den_order, 
                                                 input_signal, output_signal)
        else:
            raise ValueError("Unsupported method. Use 'scipy_minimize', 'arx', 'prony', or 'pem'.")

        num_optimized = params_optimized[:num_order + 1]
        den_optimized = np.concatenate(([1.], params_optimized[num_order + 1:]))

        system_optimized = signal.TransferFunction(num_optimized, den_optimized)

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
            w, mag, phase = signal.bode(system_optimized, w=w)
        else:
            w, mag, phase = signal.dlti(system_optimized).bode(w=w)

        # Compute additional frequency response plots
        w, h = signal.freqs(num_optimized, den_optimized, w)

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
    def mask_band(input_signal, band):
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
    
    @staticmethod
    def prepare_data(x):
        """ Convert Dask array to NumPy array if necessary. """
        if isinstance(x, da.Array):
            return x.compute()
        return np.asarray(x)

    @staticmethod
    def fit_scipy_minimize(num_order, den_order, initial_params, input_signal, 
                           output_signal, time):
        """ Fit the transfer function using scipy's minimize. """
        def fit_transfer_function(params):
            num = params[:num_order + 1]
            den = np.concatenate(([1.], params[num_order + 1:]))
            system = signal.TransferFunction(num, den)
            _, yout, _ = signal.lsim(system, U=input_signal, T=time)
            return np.sum((yout - output_signal) ** 2)

        result = minimize(fit_transfer_function, initial_params, 
                          method='Nelder-Mead')
        return result.x

    @staticmethod
    def fit_arx(num_order, den_order, input_signal, output_signal):
        """ Fit the ARX model using least squares. """
        from scipy.linalg import lstsq

        num_rows = len(input_signal) - max(num_order, den_order)
        regressor = np.zeros((num_rows, num_order + den_order + 1))

        for i in range(num_order + 1):
            regressor[:, i] = input_signal[max(num_order, den_order) - i : len(input_signal) - i]

        for j in range(1, den_order + 1):
            regressor[:, num_order + j] = -output_signal[max(num_order, den_order) - j : len(output_signal) - j]

        response = output_signal[max(num_order, den_order):]
        params_optimized, _, _, _ = lstsq(regressor, response)

        return params_optimized

    @staticmethod
    def fit_prony(num_order, den_order, output_signal):
        """ Fit the system using the Prony method. """
        N = len(output_signal)
        A = np.vstack([output_signal[i:N-den_order+i] for i in range(den_order)]).T
        b = output_signal[den_order:]
        den_params = np.linalg.lstsq(A, b, rcond=None)[0]
        den_params = np.concatenate(([1.], -den_params))

        # Fix: Compute zeros and poles from den_params
        system_poles = np.roots(den_params)
        system_zeros = np.poly1d(den_params).roots

        # Fit numerator based on the order and poles
        num_params = np.polyfit(system_poles, system_zeros, num_order)
        return np.concatenate((num_params, den_params[1:]))

    @staticmethod
    def fit_pem(num_order, den_order, input_signal, output_signal):
        """ Fit the system using the Prediction Error Method. """
        from scipy.linalg import lstsq

        num_rows = len(input_signal) - max(num_order, den_order)
        regressor = np.zeros((num_rows, num_order + den_order + 1))

        for i in range(num_order + 1):
            regressor[:, i] = input_signal[max(num_order, den_order) - i : len(input_signal) - i]

        for j in range(1, den_order + 1):
            regressor[:, num_order + j] = -output_signal[max(num_order, den_order) - j : len(output_signal) - j]

        response = output_signal[max(num_order, den_order):]
        params_optimized, _, _, _ = lstsq(regressor, response)

        return params_optimized

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
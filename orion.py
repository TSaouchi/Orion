### Path and Char manipulation
import os

### Math
import numpy as np
import scipy as spy

import Core as Orion
from DataProcessor import Processor

if __name__ == "__main__":
    # ========================= Cases configuration ============================
    cases = {
        "Zones" : [
            'BMW',
            "BWI_soft_mode",
            "BWI_hard_mode"
            ],
        "Paths" : [
            r"C:\__sandBox__\Data\pink noise\20 BMW Signale\stimuli_pink_noise_0p1_30Hz_20s\info",
            r"C:\__sandBox__\Data\pink noise\EWR_13124-846_Front_24-0021_Pink_Noise_0p4A",
            r"C:\__sandBox__\Data\pink noise\EWR_13124-846_Front_24-0021_Pink_Noise_1p6A"
        ],
        "file_name_patterns" : [
            "damper_noise_red_HA_amp<instant>",
            "Pink_Noise_HA_amp<instant>",
            "Pink_Noise_HA_amp<instant>"
            ],
        "Variables": [
            ['t', 'z', 'trigger'],
            ["Temps", "Axial Displacement"],
            ["Temps", "Axial Displacement"]
        ]
    }

    # ============================== Read inputs ===============================
    Reader = Orion.Reader
    base = []
    
    for nzone, zone in enumerate(cases["Zones"]):
        if nzone == 0:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_mat(
                                  variables = cases["Variables"][nzone],
                                  zone_name = [zone])
            
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            base.append(base_tmp)
            del base_tmp
        else:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_ascii(
                        variables = cases["Variables"][nzone],
                        zone_name = [zone])
                
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            base.append(base_tmp)
            del base_tmp
        
    # ============================= Data manipulation ==========================
    base = Processor(base).fusion()
    fft_base = Processor(base).fft(dt = 1e-2)
    
    import numpy as np
    import matplotlib.pyplot as plt

    # Sample rate and duration
    Fs = 500  # Sampling frequency (Hz)
    T = 1/Fs  # Sampling interval
    t = np.arange(0, 1, T)  # Time vector (1 second duration)

    # Create a signal with two frequencies: 50 Hz and 120 Hz
    f1 = 50  # Frequency of the first sine wave
    f2 = 120  # Frequency of the second sine wave
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    # Compute the FFT
    fft_signal = np.fft.fft(signal)
    N = len(signal)  # Number of samples
    fft_signal = fft_signal[:N // 2]  # Take the positive frequencies
    frequencies = np.fft.fftfreq(N, T)[:N // 2]  # Corresponding frequencies

    # Compute the magnitude of the FFT
    magnitude = np.abs(fft_signal)

    # Plot the time-domain signal
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.title('Time-Domain Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the frequency-domain signal (FFT)
    plt.subplot(3, 1, 2)
    plt.plot(frequencies, magnitude)
    plt.title('Frequency-Domain (FFT)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    base_tst = Orion.Base()
    base_tst.init(['toto'], ['lala'])
    base_tst[0][0].add_variable('bernard', signal)
    base_tst_fft = Processor(base_tst).fft(dt = t[1] - t[0])
    # Plot the frequency-domain signal (FFT)
    plt.subplot(3, 1, 3)
    plt.plot(base_tst_fft[0][0]['Frequency'].data, base_tst_fft[0][0]['bernard_mag'].data)
    plt.title('Frequency-Domain (FFT)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

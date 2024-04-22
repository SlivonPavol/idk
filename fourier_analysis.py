# **Popis krokov:**

# **Krok 1:**
# V tomto kroku definujeme minimálnu a maximálnu srdcovú frekvenciu, ktorú chceme odhadnúť. V tomto prípade je to 40 až 160 BPM.
#
# **Krok 2:**
# V tomto kroku vypočítame mocenskú spektrálnu hustotu (PSD) PPG signálu. PSD je metóda,
# ktorá meria, koľko energie je v signáli na každej frekvencii.
#
# **Krok 3:**
# V tomto kroku konvertujeme rozsah srdcovej frekvencie na frekvencie (Hz). To sa robí delením BPM hodnotou 60.
#
# **Krok 4:**
# V tomto kroku nájdeme index zodpovedajúci minimálnej a maximálnej frekvencii v PSD.
#
# **Krok 5:**
# V tomto kroku nájdeme odhad srdcovej frekvencie v rámci špecifikovanej oblasti.
# To sa robí nájdením frekvencie, ktorá je v strede medzi minimálnou a maximálnou frekvenciou v PSD.
#
import numpy as np
from scipy.signal import periodogram
import scipy.interpolate
import matplotlib.pyplot as plt


def estimate_heart_rate(ppg_signal, sampling_rate):
    # **Krok 1:**
    min_heart_rate_bpm = 40
    max_heart_rate_bpm = 110

    # **Krok 2:**
    frequencies, psd = periodogram(ppg_signal, fs=sampling_rate, window=None, detrend='constant', return_onesided=True, scaling='density')

    # **Krok 3:**
    min_heart_rate_hz = min_heart_rate_bpm / 60.0
    max_heart_rate_hz = max_heart_rate_bpm / 60.0

    # **Krok 4:**
    min_freq_index = np.argmax(frequencies >= min_heart_rate_hz)
    max_freq_index = np.argmax(frequencies >= max_heart_rate_hz)

    # **Krok 5:**
    estimated_heart_rate_hz = frequencies[min_freq_index + np.argmax(psd[min_freq_index:max_freq_index])]

    return estimated_heart_rate_hz

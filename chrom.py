# **Popis krokov:**

# **Krok 1:**
# V tomto kroku sa vykonáva kontrola správnosti vstupných údajov, ako je počet snímok, frekvencia snímkov a dĺžka intervalu.
#
# **Krok 2:**
# V tomto kroku sa rozdeľuje BGR signál na jednotlivé kanály - R (Red), G (Green) a B (Blue).
#
# **Krok 3:**
# V tomto kroku sa implementuje pásmový filter. Používa Butterworthov filter, ktorý je druhom filtrovania signálov.
#
# **Krok 4:**
# V tomto kroku normalizujeme a obmedzuje dĺžku kanálu (B, G, R) na základe daného intervalu.
# Funkcia upravuje kanál tak, aby sa všetky hodnoty nachádzali v intervali 0 a 1.
#
# **Krok 5:**
# V tomto kroku sa upravujú farby pokožky pod bielim svetlom, používa sa štandardná deviácia
# nakoniec ak je nejaká odchýlka v Xf a Yf tak tu sa to upravuje
#
# **Krok 6:**
# V tomto kroku sa upravuje signál a používa sa Hammingovo okno
#       **Pod krok 1:**
#       V tomto pod kroku sa vypočíta interval a potom sa nachádzajú intervalové hranice
#       **Pod krok 2:**
#       V tomto pod kroku sa vytvára Hammingovo okno
#       **Pod krok 3:**
#       V tomto kroku sa vytvoria dva zoznamy,
#       ktoré slúžia na identifikáciu rámčekov pre ľavú a pravú stranu intervalu,
#       na ktorom by sa nemalo použiť Hammingovo okno
#       **Pod krok 4:**
#       Pre každý index i v pôvodnom signáli sa kontroluje, do ktorého intervalu patrí.
#       Ak index i nie je v index_without_henning, použije sa Hammingovo okno (s vahovacím faktorom)
#       pri výpočte finálneho signálu. Inak sa použije signál bez Hammingovho okna.
#



import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import get_window


def chrom(BGR_signal, fps, interval_length):
    # **Krok 1:**
    num_frames = len(BGR_signal)
    interval_size = int(interval_length)

    if num_frames == 0:
        raise NameError('EmptyData')

    if fps < 9:
        raise NameError('WrongFPS')

    if num_frames < interval_size:
        raise NameError('NotEnoughData')

    # **Krok 2:**
    R = BGR_signal[:, 2]
    G = BGR_signal[:, 1]
    B = BGR_signal[:, 0]

    # **Krok 3:**
    def bandpass_filter(data, lowcut, highcut, sampling_frequency):
        nyquist_frequency = sampling_frequency / 2

        low_normalized = float(lowcut) / nyquist_frequency
        high_normalized = float(highcut) / nyquist_frequency

        b, a = butter(6, [low_normalized, high_normalized], btype='band')

        bandpass_data = lfilter(b, a, data)

        return bandpass_data

    # **Krok 4:**
    def normalize_and_pad_channel(channel, low_limit, high_limit):
        if low_limit < 0.0:
            num_minus = abs(low_limit)
            channel_interval = np.append(np.zeros(num_minus), channel[0:high_limit + 1])
            return channel_interval / channel_interval[num_minus:interval_size].mean()
        elif high_limit > num_frames:
            num_plus = high_limit - num_frames
            channel_interval = np.append(channel[low_limit:num_frames], np.zeros(num_plus + 1))
            return channel_interval / channel_interval[0:interval_size - num_plus - 1].mean()
        else:
            channel_interval = channel[low_limit:high_limit + 1]
            return channel_interval / channel_interval.mean()

    def S_signal_on_interval(low_limit, high_limit):
        R_interval_norm = normalize_and_pad_channel(R, low_limit, high_limit)
        G_interval_norm = normalize_and_pad_channel(G, low_limit, high_limit)
        B_interval_norm = normalize_and_pad_channel(B, low_limit, high_limit)
        # **Krok 5:**
        Xs, Ys = np.zeros(interval_size), np.zeros(interval_size)
        Xs = 3.0 * R_interval_norm - 2.0 * G_interval_norm
        Ys = 1.5 * R_interval_norm + G_interval_norm - 1.5 * B_interval_norm

        Xf = bandpass_filter(Xs, 0.5, 4.0, fps)
        Yf = bandpass_filter(Ys, 0.5, 4.0, fps)
        alpha = Xf.std() / Yf.std()
        S_before = Xf - alpha * Yf

        return S_before

    # **Krok 6:**
    #       **Pod krok 1:**
    num_intervals = int(2.0 * num_frames / interval_size + 1)

    intervals = []
    S_signal_on_intervals = []
    for i in range(num_intervals):
        interval_start = int((i - 1) * interval_size / 2.0 + 1)
        interval_end = int((i + 1) * interval_size / 2.0)
        intervals.append([interval_start, interval_end])
        S_before = S_signal_on_interval(interval_start, interval_end)
        S_signal_on_intervals.append(S_before)

    #       **Pod krok 2:**
    wh = get_window('hamming', interval_size)

    #       **Pod krok 3:**
    indices_without_hamming  = []



    # indices_without_hamming = [i for i in range(num_frames) if i not in [x for y in intervals for x in range(y[0], y[1] + 1)]]

    # Left
    for i in range(intervals[0][0], intervals[1][0], 1):
        if i >= 0:
            indices_without_hamming .append(i)
    # Right
    for i in range(intervals[len(intervals) - 2][1] + 1, intervals[len(intervals) - 1][1], 1):
        if i <= num_frames:
            indices_without_hamming .append(i)

    #       **Pod krok 4:**
    S_after = np.zeros(num_frames)
    for i in range(num_frames):
        for j, interval in enumerate(intervals):
            if interval[0] <= i <= interval[1]:
                interval_index = j
                element_on_interval = i - interval[0]
                if i not in indices_without_hamming:
                    S_after[i] += S_signal_on_intervals[interval_index][element_on_interval] * wh[element_on_interval]
                else:
                    S_after[i] += S_signal_on_intervals[interval_index][element_on_interval]


    return S_after


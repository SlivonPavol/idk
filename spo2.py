import numpy as np
from scipy.signal import firwin

def spo2(BGR_signal):
  R = BGR_signal[:, 2]
  G = BGR_signal[:, 1]
  B = BGR_signal[:, 0]
  num_frames = len(BGR_signal)
  filter_coefficients = firwin(11, cutoff=0.5, window='hamming')
  filtered_red = np.convolve(R, filter_coefficients, mode='same')
  filtered_green = np.convolve(G, filter_coefficients, mode='same')
  filtered_blue = np.convolve(B, filter_coefficients, mode='same')

  mean_red = np.mean(filtered_red)
  mean_green = np.mean(filtered_green)
  mean_blue = np.mean(filtered_blue)

  corrected_red = filtered_red / mean_red
  corrected_green = filtered_green / mean_green
  corrected_blue = filtered_blue / mean_blue

  corrected_red *= 0.7682
  corrected_green *= 0.5121
  corrected_blue *= 0.3841
  nonzero_indices_red = np.nonzero(corrected_red)[0]
  nonzero_indices_green = np.nonzero(corrected_green)[0]
  nonzero_indices_blue = np.nonzero(corrected_blue)[0]

  interpolated_corrected_red = np.interp(range(num_frames), nonzero_indices_red, corrected_red[nonzero_indices_red])
  interpolated_corrected_blue = np.interp(range(num_frames), nonzero_indices_blue, corrected_red[nonzero_indices_blue])

  red_mean = np.mean(corrected_red)
  red_std = np.std(corrected_red)
  blue_mean = np.mean(corrected_blue)
  blue_std = np.std(corrected_blue)

  spo2 = (125 - 26) * ((red_std / red_mean) / (blue_std / blue_mean))

  # Limit the spo2 value to a maximum of 100
  spo2 = min(spo2, 100)
  
  return spo2


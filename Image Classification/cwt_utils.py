import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt

"""Load CWRU vibration data and IKG experimental data."""

# DONT CHANGE
def CWRU_data(input):
    """
    Load one data per time.

    input: directory containing input data.
    """
    data = loadmat(input)
    for key, value in data.items():
        if key.endswith('DE_time'):
            df = pd.DataFrame(value, columns=[key])
            signal = df.to_numpy()
            signal = signal.flatten()
    x = np.count_nonzero(np.isnan(signal))
    if x >= 1:
        raise ValueError("nan is found in data.")
    return signal

# def read_folder(input_dir):
#     # Get a list of all files in the directory
#     files = os.listdir(input_dir)
#     for file in files:
#         # Create the full path to the file
#         file_path = os.path.join(input_dir, file)
#         # Check if it's a file (as opposed to a directory)
#         if os.path.isfile(file_path):
#             data = loadmat(file_path)
#             for key, value in data.items():
#                 if key.endswith('DE_time'):
#                     df = pd.DataFrame(value, columns=[key])
#                     signal = df.to_numpy()
#                     signal = signal.flatten()
#             return signal

def IKG_data(input):
    """
    input: directory containing input data.
    """
    header = np.arange(1, 10, 1)
    df = pd.read_csv(
        input, engine='python', delimiter='\t', skiprows=17, names=header,
        index_col=False, skipfooter=1)
    df = df[3]
    signal = df.to_numpy()
    return signal

# In progress
def segmentation(
        input, input_path, sample_rate, window_size, overlap_size,
        save_path, wavelet='morlet', dpi=100
    ):
    n = 0
    step_size = window_size - overlap_size
    f = Path(input_path)
    file_name = f.stem
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #Create segments of IMF
    for i in tqdm(range(0, len(input)-window_size, step_size)):
        n += 1
        save_name = f'{file_name}.{n}.png'
        start, end = i, (i+window_size)
        segment = input[start:end]
        synchro_cwt, norm_cwt, *_ = ssq_cwt(
            segment, wavelet=wavelet, nv=16,fs=sample_rate,
            vectorized=True, nan_checks=False, astensor=False,
            preserve_transform=False
            )
        plt.axis('off')
        plt.imshow(np.abs(synchro_cwt), aspect='auto', cmap='turbo')     
        plt.savefig(
            os.path.join(save_path, save_name), dpi=dpi,
            bbox_inches='tight', pad_inches=0
            )
        plt.close()


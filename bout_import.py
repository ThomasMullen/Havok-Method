import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from scipy.integrate import odeint
from scipy.linalg import hankel, svd, svdvals
from scipy.signal import savgol_filter
from scipy.special import binom
from scipy.optimize import curve_fit
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def print_bout_keys():
    # Print bout key
    bout_key = {1: 'ShortCS', 2: 'LongCS', 3: 'BS', 4: 'O-bend', 5: 'J-Turn', 6: 'SLC', 7: 'Slow1', 9: 'Slow2', 8: 'RT',
                10: 'LLC', 11: 'AS', 12: 'SAT', 13: 'HAT'}
    [print(item) for item in bout_key.items()]
    return


def load_filepaths(specific_bout=None, bout_spacing=None, specific_segment=None):
    # Define filepaths
    raw_data_dir = Path("/Volumes/Extreme SSD/AnalysisDynamicModeDecomposition/Data/Freezeevolution6dpfFish07")
    stim_df_path = Path("/Volumes/Extreme SSD/ExperimentFreezeEvolution/Extracted/6dpf/stimulus_key_extended_exp_2.pkl")
    print(list(raw_data_dir.glob("**/*.npy")))
    [cam_file, bout_file] = list(raw_data_dir.glob("**/*.npy"))

    # load datasets
    raw_bout = np.load(bout_file)
    bout_struct = np.load(Path("/Volumes/Extreme SSD/ExperimentFreezeEvolution/Numpy_arrays/6dpf/bout_struct_array.npy"))
    print("raw bout shape", raw_bout.shape, "bout struct shape", bout_struct.shape)

    # Print bout key
    print_bout_keys()

    # extract only fish 07
    bout_struct = bout_struct[:, 6, :, :]

    # define key variables
    pre_stimulus_duration, stimulus_duration, post_stimulus_duration, fps = 20, 5, 60, 700

    bout_struct = bout_struct.reshape(30 * 200, 12)
    bout_struct[:, 2] = bout_struct[:, 7] - (fps * pre_stimulus_duration) + bout_struct[:, 2]
    bout_struct[:, 3] = bout_struct[:, 7] - (fps * pre_stimulus_duration) + bout_struct[:, 3]

    # acquire specific bouts
    if specific_bout is not None:
        bout_struct = bout_struct[bout_struct[:, 1] == specific_bout]

    # index bout trace segments
    indexer = tuple(
        [np.s_[i:j] for (i, j) in zip(bout_struct[:, 2].astype(int), bout_struct[:, 3].astype(int))])

    # add bout spacer
    if bout_spacing is not None:
        turn_traces = np.hstack(
            [np.c_[raw_bout[:, np.r_[indexer[i]]], np.zeros((10, bout_spacing))] for i in range(len(indexer))])

    else:
        turn_traces = raw_bout[:,np.r_[indexer]]

    if specific_segment != None:
        turn_traces = turn_traces[specific_segment]

    return turn_traces






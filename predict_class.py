import os
import pickle
import librosa
import pathlib
import itertools
import numpy as np
import pandas as pd
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

with open("./storage/model_ML_cough.pkl", 'rb') as f:
    model = pickle.load(f)

with open("./storage/pca_weights.pkl", 'rb') as f:
    pca = pickle.load(f)

# Common parameters across ALL models
sampling_rate = 16000
samples_per_frame = 256
hop_length = samples_per_frame // 4
# Looping/clipping breath audio samples to the same length time_per_sample (in
# seconds). Refer eda/eda_audio_len.ipynb for stats on audio sample times.
time_per_sample_breath = 24.29  # 95th percentile value for breath.
time_per_sample_cough = 9.92    # 95th percentile value for cough.

# Parameters for SPECTROGRAM models (CNNs) -- data_spec directory
n_mels = 64

# Parameters for TRADITIONAL ML models -- data_struc directory
struc_global_features = []            # Global features need not be aggregated for an audio sample.
struc_instantaneous_features = ['rmse',
                                'zcr',
                                'sc',
                                'sb',
                                'sr',
                                'mfcc']   # Instantaneous features need to be aggregated for an audio sample.
struc_agg_funcs = [             'mean',
                                'median',
                                'rms',
                                'max',
                                'min',
                                # 'q1',
                                # 'q3',
                                # '10',
                                # '90',
                                # 'iqr',
                                # 'std',
                                # 'skew',
                                # 'kurtosis',
                                'rewm']    # Aggregation functions to use. Refer to data_struc/feature_extraction_utils.py for allowed aggregate functions.
struc_roll_percent = 0.85       # Percentage for spectral rolloff.
struc_n_mfcc = 13               # Number of MFCC coefficients to consider.

# Parameters for RECURRENT models -- data_rnn directory
rnn_instantaneous_features = ['rmse', 'zcr', 'sc', 'sb', 'sr', 'mfcc']   # Using only instantaneous features, without aggregation to preserve time component for RNN. Global features not used.
rnn_roll_percent = 0.85         # Percentage for spectral rolloff.
rnn_n_mfcc = 13                 # Number of MFCC coefficients to consider.


def load(audio_path, sampling_rate, time_per_sample):
    """
    Wrapper around librosa.load. If the audio sample is shorter than
    time_per_sample (you can set this in set_audio_params.py), it is looped back
    to time_per_sample seconds. Else if it is shorter than time_per_sample, it
    is clipped to time_per_sample seconds. Use this in place of librosa.load
    throughout this project.
    Parameters:
    audio_path (str): Absolute/relative path to audio file.
    sampling_rate (float): Number of samples to take per second (discretizing
        time).
    time_per_sample (float): Read description above for explanation.
    Returns:
    tuple: Tuple containing waveform (NumPy array), sampling_rate (float).
    """

    # Standard sampling rate is 44100 Hz.

    path = pathlib.Path(audio_path, sr=sampling_rate)
    path = os.fspath(path)

    waveform, sampling_rate = librosa.load(path)

    # Looping back if longer, clipping if shorter.
    waveform = np.resize(waveform, int(time_per_sample * sampling_rate))

    return waveform, sampling_rate


# Aggregate functions taken in the KDD paper.
agg_funcs_allowed = [
    'mean',     # Arithmetic mean
    'median',   # Median
    'rms',      # Root mean square value
    'max',      # Maximum
    'min',      # Minimum
    'q1',       # 1st quartile
    'q3',       # 3rd quartile
    'iqr',      # Interquartile range
    'std',      # Standard deviation
    'skew',     # Skewness
    'kurtosis', # Kurtosis
    'rewm'      # A custom aggregation function rms energy weighted mean, not
                # given in the KDD paper.
    # Integer values in the range [0, 100] are also allowed, representing the
    # percentile value in arr. For example, passing 95 would return the 95th
    # percentile value in arr. This too is not used in the KDD paper.
]

# Function to aggregate frame-level/instantaneous features to 1 value for the
# whole audio sample.
def aggregate(arr, agg_func, rms=None):
    if not (agg_func in agg_funcs_allowed or (agg_func.isnumeric() and (0 <= float(agg_func) <= 100))):
        raise ValueError(f'agg_func must be one among {agg_funcs_allowed} or a float in the range [0, 100] represented as a string.')
    if arr.ndim != 1 and arr.ndim != 2:
        raise ValueError(f'arr must be a tensor of rank 1.')

    if agg_func == 'mean':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.mean(arr, axis=1)
        return np.mean(arr)
    elif agg_func == 'median':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.median(arr, axis=1)
        return np.median(arr)
    elif agg_func == 'rms':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.sqrt(np.sum(arr ** 2, axis=1) / arr.shape[1])
        return np.sqrt(np.sum(arr ** 2) / len(arr))
    elif agg_func == 'max':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.max(arr, axis=1)
        return np.max(arr)
    elif agg_func == 'min':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.min(arr, axis=1)
        return np.min(arr)
    elif agg_func == 'q1':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, 25, axis=1)
        return np.percentile(arr, 25)
    elif agg_func == 'q3':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, 75, axis=1)
        return np.percentile(arr, 75)
    elif agg_func == 'iqr':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)
        return np.percentile(arr, 75) - np.percentile(arr, 25)
    elif agg_func == 'std':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.std(arr, axis=1)
        return np.std(arr)
    elif agg_func == 'skew':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return scipy.stats.skew(arr, axis=1)
        return scipy.stats.skew(arr)
    elif agg_func == 'kurtosis':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return scipy.stats.kurtosis(arr, axis=1)
        return scipy.stats.kurtosis(arr)
    elif agg_func == 'rewm':
        # Using this option requires RMS energy vector.
        if rms is None:
            raise ValueError('aggregate with agg_func as rms_energy_weighted_mean requires rms parameter.')
        # Handles case of MFCC matrix as well, which has shape (struc_n_mfcc, num_frames).
        return np.dot(arr, rms) / np.sum(rms)
    elif agg_func.isnumeric() and 0 <= float(agg_func) <= 100:
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, float(agg_func), axis=1)
        return np.percentile(arr, float(agg_func))

# INSTANTANEOUS FEATURES
# Wrappers around librosa functions that:
# 1. Use more intuitive names.
# 2. Convert optional arguments to compulsory arguments. I've spent too much
#    time debugging before just to realize later that I hadn't provided an
#    optional argument that was required to generate a desired result.
# 3. Get rid of distracting options not required by this project.
def rms_energy(waveform, samples_per_frame, hop_length):
    return librosa.feature.rms(y=waveform, frame_length=samples_per_frame, hop_length=hop_length).flatten()

def zero_crossing_rate(waveform, samples_per_frame, hop_length):
    return librosa.feature.zero_crossing_rate(waveform, frame_length=samples_per_frame, hop_length=hop_length).flatten()

def spectral_centroid(waveform, sampling_rate, samples_per_frame, hop_length):
    return librosa.feature.spectral_centroid(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length).flatten()

def spectral_bandwidth(waveform, sampling_rate, samples_per_frame, hop_length):
    return librosa.feature.spectral_bandwidth(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length).flatten()

def spectral_rolloff(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent):
    return librosa.feature.spectral_rolloff(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, roll_percent=roll_percent).flatten()

def mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc):
    return librosa.feature.mfcc(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mfcc=n_mfcc)

def dmfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc):
    mfcc = librosa.feature.mfcc(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mfcc=n_mfcc)
    return librosa.feature.delta(mfcc)

def d2mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc):
    mfcc = librosa.feature.mfcc(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mfcc=n_mfcc)
    return librosa.feature.delta(mfcc, order=2)

# AGGREGATE INSTANTANEOUS FEATURES
# Note that aggregate function 'rewm' requires slightly different treatment (it
# requires the root mean square energies rms to be passed to the aggregate
# function), because of the definition of 'rewm'.
def rms_energy_agg(waveform, samples_per_frame, hop_length, agg_func='95', rms=None):
    """ Returns aggregate of framewise RMS energies, for an audio clip. """
    rms_energies = rms_energy(waveform, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate rms energy.
        # I don't know if it makes sense to weight rms energy with rms energy
        # to aggregate it, it'd just be squaring the rms energies over the
        # frames, and taking their mean. Keeping it for the sake of consistency.
        # If required, it can be removed from the csv files.
        return aggregate(rms_energies, agg_func, rms=rms)
    return aggregate(rms_energies, agg_func)

def zero_crossing_rate_agg(waveform, samples_per_frame, hop_length, agg_func, rms=None):
    """ Returns aggregate of framewise zero crossing rates, for an audio clip. """

    zcr = zero_crossing_rate(waveform, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(zcr, agg_func, rms=rms)
    return aggregate(zcr, agg_func)

def spectral_centroid_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=None):
    """ Returns aggregate of spectral centroids, for an audio clip. """

    spec_centroids = spectral_centroid(waveform, sampling_rate, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(spec_centroids, agg_func, rms=rms)
    return aggregate(spec_centroids, agg_func)

def spectral_bandwidth_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=None):
    """ Returns aggregate of framewise spectral bandwidths, for an audio clip. """

    spec_bws = spectral_bandwidth(waveform, sampling_rate, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(spec_bws, agg_func, rms=rms)
    return aggregate(spec_bws, agg_func)

def spectral_rolloff_agg(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent, agg_func, rms=None):
    """ Returns aggregate of framewise spectral rolloffs, for an audio clip. """

    spec_rolloffs = spectral_rolloff(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(spec_rolloffs, agg_func, rms=rms)
    return aggregate(spec_rolloffs, agg_func)

def mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func, rms=None):
    """ Returns aggregate across time axis (axis=1) of MFCCs, for an audio clip. """

    mfccs = mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(mfccs, agg_func, rms=rms)
    return aggregate(mfccs, agg_func)

def dmfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func, rms=None):
    """ Returns aggregate across time axis (axis=1) of derivative of MFCCs, for an audio clip. """

    dmfccs = dmfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(dmfccs, agg_func, rms=rms)
    return aggregate(dmfccs, agg_func)

def d2mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func, rms=None):
    """ Returns aggregate across time axis (axis=1) of second derivative of MFCCs, for an audio clip. """

    d2mfccs = d2mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(d2mfccs, agg_func, rms=rms)
    return aggregate(d2mfccs, agg_func)



def generate_feature_names():
    """
    Generates names of feature columns to be used in the training and test dataframes from the feature names mentioned in set_audio_params.py.
    Parameters:
    None
    Returns:
    list: List of names of feature columns to be used in the training and test dataframes.
    """

    # MFCC features are instantaneous features.
    if 'mfcc' in struc_instantaneous_features:
        mfcc_features = ['mfcc' + str(i) for i in range(1, struc_n_mfcc + 1)]
        struc_instantaneous_features.extend(mfcc_features)

    # Removing the dummy literal 'mfcc' which stood for all coefficients from mfcc0
    # to mfcc<struc_n_mfcc>, as we have already handled the mfcc features above.
    if 'mfcc' in struc_instantaneous_features:
        struc_instantaneous_features.remove('mfcc')

    # Generating feature names for struc_instantaneous_features x struc_agg_funcs.
    struc_instantaneous_features_agg = [instantaneous_feature + '_' + str(agg_func) for instantaneous_feature, agg_func in itertools.product(struc_instantaneous_features, struc_agg_funcs)]

    # All features
    features = struc_instantaneous_features_agg + struc_global_features

    return features

def generate_feature_row(orig_df, filename, class_, waveform, sampling_rate, samples_per_frame, hop_length):
    """ Returns a row of features as a Pandas DataFrame. """

    # row_df will be appended to orig_df, hence must have same columns.
    row_df = pd.DataFrame(columns=orig_df.columns)

    # Pre-calculating rms energy if agg_func is rewm. Remember that rewm
    # requires slightly different treatment than the other aggregate functions.
    # Passing it as a parameter regardless of the aggregate functions, if it is
    # not rewm, rms is ignored.
    rms = rms_energy(waveform, samples_per_frame, hop_length)

    for feature in row_df.columns:
        # Instantaneous aggregate features contain '_' as substring.
        if '_' in feature:
            feature_name, agg_func = feature.split('_')

            if feature_name == 'rmse':
                row_df.loc[filename, feature_name + '_' + agg_func] = rms_energy_agg(waveform, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'zcr':
                row_df.loc[filename, feature_name + '_' + agg_func] = zero_crossing_rate_agg(waveform, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'sc':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_centroid_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'sb':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_bandwidth_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'sr':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_rolloff_agg(waveform, sampling_rate, samples_per_frame, hop_length, struc_roll_percent, agg_func, rms=rms)
            # Handling MFCC separately.
            elif 'mfcc' in feature_name:
                continue

        elif feature == 'target':
            row_df.loc[filename, feature] = class_

        # Global features.
        else:
            # No global features yet.
            pass

    # Handling MFCC features separately.
    mfcc_features = [feature for feature in orig_df.columns.tolist() if 'mfcc' in feature]
    # Sanity check that max mfcc coefficient in mfcc_features is same as struc_n_mfcc from set_audio_params.py.
    assert struc_n_mfcc == max([int(mfcc_feature.split('_')[0][4:]) for mfcc_feature in mfcc_features])
    mfcc_struc_agg_funcs = set([mfcc_feature.split('_')[1] for mfcc_feature in mfcc_features])

    for agg_func in struc_agg_funcs:
        # Vector of mfcc_max_coef number of MFCCs.
        mfcc_vec = mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, struc_n_mfcc, agg_func, rms=rms)
        for i in range(1, struc_n_mfcc + 1):
            row_df.loc[filename, 'mfcc' + str(i) + '_' + str(agg_func)] = mfcc_vec[i - 1]

    return row_df

def add_is_cough_symptom(filename):
    """ Returns whether cough is a symptom, using the filename (index of the dataframe). """

    # Filename convention comes useful here.
    # Filename convention comes useful here.
    is_symptom = filename.split('_')[2]

    if 'with' in is_symptom:
        return 1
    elif 'no' in is_symptom:
        return 0
    else:
        print('Make sure the filename convention is followed.')
        sys.exit(1)


def extract_features(filepath, audio_type):
    """
    Extracts the same features as used to train the model (refer set_audio_params.py or data_struc/train.csv) from the input audio sample at filepath.
    Wrapper around generate_feature_row from generate_features.py.
    Parameters:
    filepath (str): Path to audio file.
    audio_type (str): Can be 'breath' or 'cough'.
    Returns:
    pandas.DataFrame: Row of features as a Pandas DataFrame.
    """

    features = generate_feature_names()

    # No target column as we need to predict.
    orig_df = pd.DataFrame(columns=features + ['filename'])
    orig_df = orig_df.set_index('filename')

    if audio_type == 'breath':
        waveform, _ = load(filepath, sampling_rate, time_per_sample_breath)
    elif audio_type == 'cough':
        waveform, _ = load(filepath, sampling_rate, time_per_sample_cough)

    # class_ parameter is used only if feature_name is 'target'. However, we don't pass 'target' as the name of a feature column, so passing a dummy value for class_.
    filename = os.path.basename(filepath)
    row_df = generate_feature_row(orig_df, filename, -1, waveform, sampling_rate, samples_per_frame, hop_length)

    return row_df


def predict(filepath, audio_type, is_cough_symptom):
    """
    Predicts whether an audio sample is normal or covid.
    Parameters:
    filepath (str): Path to audio file.
    classifier (sklearn.base.BaseEstimator): scikit-learn estimator object with predict method.
    audio_type (str): Can be 'breath' or 'cough'.
    is_cough_symptom (boolean): Corresponds to is_cough_symptom in train.csv, whether the user reported cough as a symptom or not.
    Returns:
    int: Output of predict method of classifier (by default 0 for normal, 1 for covid).
    """

    row_df = extract_features(filepath, audio_type)
    row_df['is_cough_symptom'] = is_cough_symptom

    row_df = pca.transform(row_df)

    # row_df.to_csv('tmp.csv')

    prediction = model.predict(row_df)[0]

    if prediction == 0:
        return 'covid'
    else:
        return 'normal'
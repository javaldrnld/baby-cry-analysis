import librosa
import numpy as np

max_length = 100


def preemphasis_filter(signal, alpha=0.97):
    """
    Apply a pre-emphasis filter to the input signal.

    This function applies a first-order high-pass filter to the input signal to emphasize higher frequencies.
    The filter is defined as: y[t] = x[t] - alpha * x[t-1]

    Parameters:
    signal (numpy.ndarray): The input signal to be filtered.
    alpha (float, optional): The pre-emphasis coefficient. Default is 0.97.

    Returns:
    numpy.ndarray: The filtered signal with pre-emphasis applied.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# Frame the signal into 25 ms frame and 10 ms frame shift
def frame_signal(signal, frame_length, frame_stride):
    """
    Splits the input signal into overlapping frames.

    Parameters:
    signal (array-like): The input signal to be framed.
    frame_length (int): The length of each frame.
    frame_stride (int): The stride between consecutive frames.

    Returns:
    numpy.ndarray: A 2D array where each row is a frame of the signal.
    """

    signal_length = len(signal)
    frame_step = int(frame_stride)
    frame_length = int(frame_length)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    return frames


# Apply Hamming Windows and Compute the power spectrum
def power_spectrum(frames, NFFT=512):
    """
    Compute the power spectrum of the given frames.

    Parameters:
    frames (numpy.ndarray): A 2D array where each row is a frame of audio data.
    NFFT (int, optional): The number of points in the FFT. Default is 512.

    Returns:
    numpy.ndarray: A 2D array where each row is the power spectrum of the corresponding frame.
    """

    frames *= np.hamming(frames.shape[1])
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames**2)
    return pow_frames


# Apply Mel Filterbank
def mel_filterbank(spectrum, num_filters=40, sampling_rate=22050, n_fft=512):
    """
    Apply a Mel filterbank to the given spectrum.

    Parameters:
    spectrum (numpy.ndarray): The input spectrum to which the Mel filterbank will be applied.
    num_filters (int, optional): The number of Mel filters to use. Default is 40.
    sampling_rate (int, optional): The sampling rate of the audio signal. Default is 22050 Hz.
    n_fft (int, optional): The number of FFT components. Default is 512.

    Returns:
    numpy.ndarray: The spectrum after applying the Mel filterbank.
    """
    mel_filterbank = librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_filters
    )
    return np.dot(mel_filterbank, spectrum.T).T


# Compute MFCCS
def mfcc(
    signal,
    sampling_rate=22050,
    frame_length=512,
    frame_shift=256,
    num_mfcc=13,
    n_mels=40,
):
    """
    Compute the Mel-frequency cepstral coefficients (MFCCs) from an audio signal.
    Parameters:
    signal (numpy.ndarray): The input audio signal.
    sampling_rate (int, optional): The sampling rate of the audio signal. Default is 22050.
    frame_length (int, optional): The length of each frame in samples. Default is 512.
    frame_shift (int, optional): The number of samples to shift for the next frame. Default is 256.
    num_mfcc (int, optional): The number of MFCCs to return. Default is 13.
    n_mels (int, optional): The number of Mel bands to generate. Default is 40.
    Returns:
    numpy.ndarray: The flattened array of MFCCs with a fixed length.
    """

    emphasized_signal = preemphasis_filter(signal)
    framed_signal = frame_signal(emphasized_signal, frame_length, frame_shift)
    spectrum = power_spectrum(framed_signal)
    mel_spectrum = mel_filterbank(
        spectrum, num_filters=n_mels, sampling_rate=sampling_rate, n_fft=512
    )
    mfccs = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spectrum), n_mfcc=num_mfcc, n_mels=n_mels
    )

    # Ensure the MFCCs have a fixed length
    if mfccs.shape[1] < max_length:
        mfccs_padded = np.pad(
            mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode="constant"
        )
    elif mfccs.shape[1] > max_length:
        mfccs_padded = mfccs[:, :max_length]
    else:
        mfccs_padded = mfccs

    return mfccs_padded.flatten()


# Feature Extraction: MFCC simplified
def extract_features(audio_part, sigr=22050, n_mfcc=13, hop_length=None, n_fft=None):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio segment with optional augmentation.

    Parameters:
    audio_part (AudioSegment): The audio segment to process.
    sigr (int, optional): The sample rate of the audio. Default is 22050.
    n_mfcc (int, optional): The number of MFCCs to return. Default is 13.
    hop_length (int, optional): The number of samples between successive frames. If None, defaults to 10ms of `sigr`.
    n_fft (int, optional): The length of the FFT window. If None, defaults to 25ms of `sigr`.

    Returns:
    np.ndarray: A 2D array of shape (time_steps, n_mfcc) containing the MFCCs.
    """

    if hop_length is None:
        hop_length = int(0.010 * sigr)
    if n_fft is None:
        n_fft = int(0.025 * sigr)
    audio_part = np.array(audio_part.get_array_of_samples(), dtype=np.float32)
    mfccs = librosa.feature.mfcc(
        y=audio_part, sr=sigr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
    )
    return mfccs.T  # Transpose to have time steps as rows

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import math
import os
import requests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
import librosa.display as display
import librosa.feature
from scipy.signal import resample

###############################################################################################################

def extract_audio(filename):
    """
    Extract audio given the filename (.wav, .flac, etc format)
    """

    audio, rate = sf.read(filename, always_2d=True)
    audio = np.reshape(audio, (1, -1))
    audio = audio[0]
    time = np.linspace(0, len(audio)/rate, len(audio), endpoint=False)
    return audio, time, rate

def generate_spec(audio_sequence, rate, n_fft=2048, hop_length=512):
    """
    Generate spectrogram using librosa
    audio_sequence: list representing waveform
    rate: sampling rate (16000 for all LibriSpeech audios)
    nfft and hop_length: stft parameters
    """
    S = librosa.feature.melspectrogram(audio_sequence, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmin=20,
                                       fmax=8300)
    log_spectra = librosa.power_to_db(S, ref=np.mean, top_db=80)
    return log_spectra

def reconstruct_wave(spec, rate=16000, normalize_data=False):
    """
    Reconstruct waveform
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    power = librosa.db_to_power(spec, ref=5.0)
    audio = librosa.feature.inverse.mel_to_audio(power, sr=rate, n_fft=2048, hop_length=512)
    out_audio = audio / np.max(audio) if normalize_data else audio
    return out_audio

def normalize(spec, eps=1e-6):
    """
    Normalize spectrogram with zero mean and unitary variance
    spec: spectrogram generated using Librosa
    """

    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    return spec_norm, (mean, std)

def minmax_scaler(spec):
    """
    min max scaler over spectrogram
    """
    spec_max = np.max(spec)
    spec_min = np.min(spec)

    return (spec-spec_min)/(spec_max - spec_min), (spec_max, spec_min)

def linear_scaler(spec):
    """
    linear scaler over spectrogram
    min value -> -1 and max value -> 1
    """
    spec_max = np.max(spec)
    spec_min = np.min(spec)
    m = 2/(spec_max-spec_min)
    n = (spec_max + spec_min)/(spec_min-spec_max)

    return m*spec + n, (m, n)

def split_specgram(example, clean_example, frames = 11):
    """
    Split specgram in groups of frames, the purpose is prepare data for the LSTM model input

    example: reverberant spectrogram
    clean_example: clean or target spectrogram

    return data input to the LSTM model and targets
    """
    clean_spec = clean_example[0, :, :]
    rev_spec = example[0, :, :]

    n, m = clean_spec.shape

    targets = torch.zeros((m-frames+1, n))
    data = torch.zeros((m-frames+1, n*frames))
  
    idx_target = frames//2
    for i in range(m-frames+1):
        try:
            targets[i, :] = clean_spec[:, idx_target]
            data[i, :] = torch.reshape(rev_spec[:, i:i+frames], (1, -1))[0, :]
            idx_target += 1
        except (IndexError):
            pass
    return data, targets

def split_realdata(example, frames = 11):
    
    """
    Split 1 specgram in groups of frames, the purpose is prepare data for the LSTM and MLP model input

    example: reverberant ''real'' (not simulated) spectrogram

    return data input to the LSTM or MLP model 
    """
  
    rev_spec = example[0, :, :]
    n, m = rev_spec.shape
    data = torch.zeros((m-frames+1, n*frames))
    for i in range(m-frames+1):
        data[i, :] = torch.reshape(rev_spec[:, i:i+frames], (1, -1))[0, :]
    return data

def prepare_data(X, y, display = False):

    """
    Use split_specgram to split all specgrams
    X: tensor containing reverberant spectrograms
    y: tensor containing target spectrograms
    """

    data0, target0 = split_specgram(X[0, :, :, :], y[0, :, :, :])

    total_data = data0.cuda()
    targets = target0.cuda()
  
    for i in range(1, X.shape[0]):
           if display: 
               print("Specgram n°" + str(i)) 

           data_i, target_i = split_specgram(X[i, :, :, :], y[i, :, :, :])
           total_data = torch.cat((total_data, data_i.cuda()), 0)
           targets = torch.cat((targets, target_i.cuda()), 0)

    return  total_data, targets


def split_for_supression(rev_tensor, target_tensor):
    """
    Given reverberant and target tensor with shape (#examples, 1, 128, 340)
    return tensors with the same information, but with shape (#examples*340, 128)
    """
    rev_transform = torch.tensor([])
    target_transform = torch.tensor([])

    for example in range(rev_tensor.shape[0]):
        rev_transform = torch.cat((rev_transform, rev_tensor[example, 0, :, :].T))
    
    if (target_tensor!=None):
        for example in range(target_tensor.shape[0]):
            target_transform = torch.cat((target_transform, target_tensor[example, 0, :, :].T))
  
    return rev_transform, target_transform

def normalize_per_frame(spec_transpose):
    """
    Normalize over spectrogram rows
    """
    means = []
    stds = []
    norm_spec = torch.zeros(spec_transpose.shape)

    for spec_row in range(norm_spec.shape[0]):
        current_mean = spec_transpose[spec_row, :].mean()
        current_std = spec_transpose[spec_row, :].std()
        means.append(current_mean)
        stds.append(current_std)
        norm_spec[spec_row, :] = (spec_transpose[spec_row, :]- current_mean)/(current_std+1e-6) 
  
    return norm_spec, (means, stds)

def denormalize_per_frame(norm_spec_transpose, means, stds):
    """
    denormalize row by row using means and stds given by normalize_per_frame
    """
    denorm_spec = torch.zeros(norm_spec_transpose.shape)

    for spec_row in range(norm_spec_transpose.shape[0]):
        denorm_spec[spec_row, :] = (norm_spec_transpose[spec_row, :])*(stds[spec_row] + 1e-6) + means[spec_row]
    
    return denorm_spec.T


#################################
# reverberation utils
#################################

def zero_pad(x, k):
    """
    add k zeros to x signal
    """
    return np.append(x, np.zeros(k))


def awgn(signal, regsnr):
    """
    add random noise to signal
    regsnr: signal to noise ratio
    """
    sigpower = sum([math.pow(abs(signal[i]), 2) for i in range(len(signal))])
    sigpower = sigpower / len(signal)
    noisepower = sigpower / (math.pow(10, regsnr / 10))
    sample = np.random.normal(0, 1, len(signal))
    noise = math.sqrt(noisepower) * sample
    return noise


def discrete_conv(x, h, x_fs, h_fs, snr=30, aug_factor=1):
    """
    Convolution using fft
    x: speech waveform
    h: RIR waveform
    x_fs: speech signal sampling rate (if is not 16000 the signal will be resampled)
    h_fs: RIR signal sampling rate (if is not 16000 the signal will be resampled)

    Based on https://github.com/vtolani95/convolution/blob/master/reverb.py
    """

    numSamples_h = round(len(h) / h_fs * 16000)
    numSamples_x = round(len(x) / x_fs * 16000)

    if h_fs != 16000:
        h = resample(h, numSamples_h) # resample RIR

    if x_fs != 16000:
        x = resample(x, numSamples_x) # resample speech signal

    L, P = len(x), len(h)
    h_zp = zero_pad(h, L - 1)
    x_zp = zero_pad(x, P - 1)
    X = np.fft.fft(x_zp)
    output = np.fft.ifft(X * np.fft.fft(h_zp)).real
    output = aug_factor * output + x_zp
    output = output + awgn(output, snr)
    return output

###################################
#plot utils
###################################

def graph_spec(spec, rate=16000, title=False):
    """
    plot spectrogram
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    plt.figure()
    display.specshow(spec, sr=rate, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    if (title):
        plt.title('Log-Power spectrogram')
    plt.tight_layout()

def plot_time_wave(audio, rate=16000):
    """
    plot waveform given speech audio
    audio: array containing waveform
    rate: sampling rate

    """
    time = np.linspace(0, len(audio)/rate, len(audio), endpoint=False)
    plt.figure()
    plt.plot(time, audio)
    plt.xlabel("Time (secs)")
    plt.ylabel("Power")

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#rir_raw, sample_rate = torchaudio.load("MediumRm.wav")
#Audio(rir_raw, rate=sample_rate)
#rir = rir_raw
##rir2 = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]   # extracting the IR---not needed with already raw files!
#
#rir = rir / torch.norm(rir, p=2)     # Normalization
#RIR = torch.flip(rir, [1])
#
##rir2 = rir2 / torch.norm(rir2, p=2)     # Normalization
##RIR2 = torch.flip(rir2, [1])
#########################################################################

dry_sample = "origin.wav"
wet_sample = "temp.wav"

waveform1, sample_rate1= torchaudio.load(dry_sample)
waveform2, sample_rate2= torchaudio.load(wet_sample)

n_fft = 2048
win_length = 2048
hop_length = 512

# define transformation
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)
# Perform transformation
spec1 = spectrogram(waveform1)
spec2 = spectrogram(waveform2)

#print_stats(spec)
plot_spectrogram(spec1[0], title='dry')
plt.savefig("output1.jpg") #save as jpg
plot_spectrogram(spec2[0], title='wet')
plt.savefig("output2.jpg") #save as jpg

#for filename in os.listdir(dry_dir):
#	dry_sample = os.path.join(dry_dir, filename)
#    count = 1
#	if os.path.isfile(dry_sample) and count < 4:
#        speech, sample_rate_original = torchaudio.load(dry_sample)
#        speech_ = torch.nn.functional.pad(speech, (RIR.shape[1] - 1, 0))
#        #speech_2 = torch.nn.functional.pad(speech, (RIR2.shape[1] - 1, 0))
#        doubled_speech_ = speech_.repeat(2,1)
#        #doubled_speech_2 = speech_2.repeat(2,1)
#        augmented = torch.nn.functional.conv1d(doubled_speech_[None, ...], RIR[None, ...])[0]
#        #augmented2 = torch.nn.functional.conv1d(doubled_speech_2[None, ...], RIR2[None, ...])[0]
#        wet_sample_name = filename
#        torchaudio.save(wet_sample_name, augmented, sample_rate_original)
#        #torchaudio.save('temp2.wav', augmented2, sample_rate_original)
#        count += 1
#        if count == 3
#            break
#
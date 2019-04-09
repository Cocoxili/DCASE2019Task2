
from util import *
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from config import *


def get_wavelist(curated=True, noisy=True, test=True):
    # usage: 0 for test audio, 1 for curated label, 2 for noisy label.
    df = pd.DataFrame(columns=['fname', 'usage'])

    if curated:
        train_curated_dir = '../input/train_curated'
        waves_train_curated = sorted(os.listdir(train_curated_dir))
        for fn in waves_train_curated:
            df = df.append({'fname': fn, 'usage': 1}, ignore_index=True)
        print("train_curated: ", len(waves_train_curated))

    if noisy:
        train_noisy_dir = '../input/train_noisy'
        waves_train_noisy = sorted(os.listdir(train_noisy_dir))
        for fn in waves_train_noisy:
            df = df.append({'fname': fn, 'usage': 2}, ignore_index=True)
        print("train_noisy: ", len(waves_train_noisy))

    if test:
        test_dir = '../input/test'
        waves_test = sorted(os.listdir(test_dir))
        for fn in waves_test:
            df = df.append({'fname': fn, 'usage': 0}, ignore_index=True)
        print("test: ", len(waves_test))

    df.set_index('fname', inplace=True)
    df.to_csv('./wavelist.csv')


def wav_to_pickle(wavelist):
    df = pd.read_csv(wavelist)
    pool = Pool(10)
    pool.map(tsfm_wave, df.iterrows())


def wav_to_logmel(wavelist):
    df = pd.read_csv(wavelist)
    pool = Pool(10)
    pool.map(tsfm_logmel, df.iterrows())


def wav_to_mfcc(wavelist):
    df = pd.read_csv(wavelist)
    pool = Pool(10)
    pool.map(tsfm_mfcc, df.iterrows())


def tsfm_wave(row):
    sr = config_wave.sampling_rate
    item = row[1]
    p_name = os.path.join('../input/features/wave-44100',
                          os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        if item['usage'] == 0:
            file_path = os.path.join('../input/test/', item['fname'])
        elif item['usage'] == 1:
            file_path = os.path.join('../input/train_curated/', item['fname'])
        elif item['usage'] == 2:
            file_path = os.path.join('../input/train_noisy/', item['fname'])

    print(row[0], file_path)
    data, _ = librosa.core.load(file_path, sr=sr, res_type='kaiser_best')
    save_data(p_name, data)


def tsfm_logmel(row):

    item = row[1]
    p_name = os.path.join('../input/features/logmel+delta_w80_s10_m64',
                          os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        if item['usage'] == 0:
            file_path = os.path.join('../input/test/', item['fname'])
        elif item['usage'] == 1:
            file_path = os.path.join('../input/train_curated/', item['fname'])
        elif item['usage'] == 2:
            file_path = os.path.join('../input/train_noisy/', item['fname'])

        data, sr = librosa.load(file_path, config.sampling_rate)

        # some audio file is empty, fill logmel with 0.
        if len(data) == 0:
            print("empty file:", file_path)
            logmel = np.zeros((config.n_mels, 150))
            feats = np.stack((logmel, logmel, logmel))
        else:
            melspec = librosa.feature.melspectrogram(data, sr,
                                                     n_fft=config.n_fft, hop_length=config.hop_length,
                                                     n_mels=config.n_mels)

            logmel = librosa.core.power_to_db(melspec)

            delta = librosa.feature.delta(logmel)
            accelerate = librosa.feature.delta(logmel, order=2)

            feats = np.stack((logmel, delta, accelerate)) #(3, 64, xx)

        save_data(p_name, feats)


def tsfm_mfcc(row):

    item = row[1]

    p_name = os.path.join('../mfcc+delta_w80_s10_m64', os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        if item['usage'] == 0:
            file_path = os.path.join('../input/test/', item['fname'])
        elif item['usage'] == 1:
            file_path = os.path.join('../input/train_curated/', item['fname'])
        elif item['usage'] == 2:
            file_path = os.path.join('../input/train_noisy/', item['fname'])

        data, sr = librosa.load(file_path, config.sampling_rate)

        # some audio file is empty, fill logmel with 0.
        if len(data) == 0:
            print("empty file:", file_path)
            mfcc = np.zeros((config.n_mels, 150))
            feats = np.stack((mfcc, mfcc, mfcc))
        else:
            mfcc = librosa.feature.mfcc(data, sr,
                                        n_fft=config.n_fft,
                                        hop_length=config.hop_length,
                                        n_mfcc=config.n_mels)
            delta = librosa.feature.delta(mfcc)
            accelerate = librosa.feature.delta(mfcc, order=2)

            feats = np.stack((mfcc, delta, accelerate)) #(3, 64, xx)

        save_data(p_name, feats)


if __name__ == '__main__':
    make_dirs()
    config = Config(sampling_rate=22050, n_mels=64, frame_weigth=80, frame_shift=10)
    config_wave = Config(sampling_rate=44100, n_mels=64, frame_weigth=40, frame_shift=10)
    # get_wavelist(curated=True, noisy=False, test=True)

    # what kind of feature to extract? wave, logmel or MFCC?
    # wav_to_pickle('wavelist.csv')
    wav_to_logmel('wavelist.csv')
    # wav_to_mfcc('wavelist.csv')

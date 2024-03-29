
from util import *
from tqdm import tqdm
from multiprocessing import Pool


def wave_to_logmel(subset, wavelist):
    X = {}
    if subset == 'test':
        sub_dir = config.test_dir
    elif subset == 'train_curated':
        sub_dir = config.train_curated_dir
    elif subset == 'train_noisy':
        sub_dir = config.train_noisy_dir

    for i, item in tqdm(wavelist.iterrows()):
        file_path = os.path.join(sub_dir, item['fname'])

        data, sr = librosa.load(file_path, config.sampling_rate)

        data, _ = librosa.effects.trim(data)

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
            logmel = logmel.astype(np.float32)  # float32 make the feature much small

            if logmel.shape[1] < 10:
                print("Too short audio:", file_path)
                delta = np.zeros_like(logmel)
                accelerate = np.zeros_like(logmel)
            else:
                delta = librosa.feature.delta(logmel)
                accelerate = librosa.feature.delta(logmel, order=2)
            feats = np.stack((logmel, delta, accelerate))

        X[item['fname']] = feats
    return X


def wave_to_pkl(subset, wavelist):
    X = {}
    if subset == 'test':
        sub_dir = config.test_dir
    elif subset == 'train_curated':
        sub_dir = config.train_curated_dir
    elif subset == 'train_noisy':
        sub_dir = config.train_noisy_dir

    for i, item in tqdm(wavelist.iterrows()):
        file_path = os.path.join(sub_dir, item['fname'])

        data, _ = librosa.load(file_path, config.sampling_rate, res_type='kaiser_best')

        # some audio file is empty, fill logmel with 0.
        if len(data) == 0:
            print("empty file:", file_path)
        else:
            X[item['fname']] = data
    return X


def calculate_mean_and_std(X, channel):
    X = list(X.values())
    means = []
    stds = []
    for channel in range(channel):
        A = np.array([0])
        for i in range(0, len(X), 3):
            A = np.append(A, X[i][channel].flatten())
            print(i, A.shape)
        means.append(np.mean(A))
        stds.append(np.std(A))
    print(means, stds)
    return means, stds


def logmel_normalize(X, means, stds):
    for key in X.keys():
        X[key] = ((X[key] - means[:, None, None]) / stds[:, None, None]).astype(np.float32)
    return X


def get_wavelist(subset='test'):
    if subset == 'train_curated':
        wavelist = pd.read_csv(config.CSV_TRAIN_CURATED)
    elif subset == 'test':
        wavelist = pd.read_csv(config.CSV_SBM)
    elif subset == 'train_noisy':
        wavelist = pd.read_csv(config.CSV_TRAIN_NOISY)
    if config.debug:
        wavelist = wavelist[:20]
    return wavelist


def get_logmel_feature(subset):
    end = time.time()
    wavelist = get_wavelist(subset)
    print(len(wavelist))
    X = wave_to_logmel(subset, wavelist)
    print('Time of data transformation: ', time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - end)))
    return X


def get_wave_feature(subset):
    end = time.time()
    wavelist = get_wavelist(subset)
    print(len(wavelist))
    X = wave_to_pkl(subset, wavelist)
    print('Time of data transformation: ', time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - end)))
    return X


def wav_to_logmel(wavelist):
    df = pd.read_csv(wavelist)
    pool = Pool(10)
    pool.map(tsfm_logmel, df.iterrows())


def tsfm_logmel(row):

    item = row[1]
    p_name = os.path.join(config.features_dir, 'train_curated', os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        file_path = os.path.join(config.train_curated_dir, item['fname'])

        data, sr = librosa.load(file_path, config.sampling_rate)

        data, _ = librosa.effects.trim(data)

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
            logmel = logmel.astype(np.float32)  # float32 make the feature much small

            if logmel.shape[1] < 10:
                print("Too short audio:", file_path)
                delta = np.zeros_like(logmel)
                accelerate = np.zeros_like(logmel)
            else:
                delta = librosa.feature.delta(logmel)
                accelerate = librosa.feature.delta(logmel, order=2)
            feats = np.stack((logmel, delta, accelerate))

            feats = ((feats - means[:, None, None]) / stds[:, None, None]).astype(np.float32)
        save_data(p_name, feats)


if __name__ == '__main__':
    config = Config(
                    # csv_train_noisy='./trn_noisy_best50s.csv',
                    csv_train_noisy='../../../input/train_noisy.csv',
                    sampling_rate=44100,
                    n_mels=128,
                    frame_weigth=100,
                    frame_shift=5,
                    features_dir="../../../features/logmel_w100_s5_m128_trim_norm",
                    debug=False)

    # for logmel_w100_s10_m128_trim
    means = np.array([-26.6642, -0.0131, -0.0028])
    stds = np.array([20.1533, 1.0592, 0.4690])

    # for logmel_w100_s10_m128_trim noisy50
    # means_noisy = np.array([-19.0795, -0.0002, -0.00026])
    # stds_noisy = np.array([21.4800, 0.5806, 0.1764])

    # for logmel_1c_w100_s5_m256_trim
    # means = np.array([-26.5654])
    # stds = np.array([19.7549])

    # X_test = get_logmel_feature(subset='test')
    # X_test = logmel_normalize(X_test, means, stds)
    # save_data(os.path.join(config.features_dir, 'test.pkl'), X_test)

    # X_train_curated = get_logmel_feature(subset='train_curated')
    # X_train_curated = logmel_normalize(X_train_curated, means, stds)
    # save_data(os.path.join(config.features_dir, 'train_curated.pkl'), X_train_curated)

    # X_train_noisy = get_logmel_feature(subset='train_noisy')
    # X_train_noisy = logmel_normalize(X_train_noisy, means_noisy, stds_noisy)
    # save_data(os.path.join(config.features_dir, 'train_noisy_all.pkl'), X_train_noisy)
    # print(pd.read_csv(config.CSV_TRAIN_NOISY))
    # wav_to_logmel(config.CSV_TRAIN_CURATED)
    # wav_to_logmel(config.CSV_TRAIN_NOISY)


    # config = Config(
    #                 csv_train_noisy='./trn_noisy_best50s.csv',
    #                 # csv_train_noisy='../input/train_noisy.csv',
    #                 sampling_rate=44100,
    #                 features_dir="../../../features/wave_sr44100")
    #
    # X_test = get_wave_feature(subset='test')
    # save_data(os.path.join(config.features_dir, 'test.pkl'), X_test)
    #
    # X_train_curated = get_wave_feature(subset='train_curated')
    # save_data(os.path.join(config.features_dir, 'train_curated.pkl'), X_train_curated)
    #
    # X_train_noisy = get_wave_feature(subset='train_noisy')
    # save_data(os.path.join(config.features_dir, 'train_noisy.pkl'), X_train_noisy)

    # X = load_data('/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_w100_s5_m128_trim_norm/train_noisy50.pkl')
    # X1 = load_data('/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_1c_w100_s5_m256_trim/test.pkl')
    # X.update(X1)
    # print(len(X))
    # calculate_mean_and_std(X, channel=3)
    # X = load_data(
    #     '/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_w100_s5_m128_trim_norm/test.pkl')
    # print(X['001ed5f1.wav'])
    # print(X['001ed5f1.wav'].shape)
    # X = load_data('/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_w100_s5_m128_trim_norm/train_curated/0019ef41.pkl')
    # X2 = load_data('/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_w100_s5_m128_trim_norm/train_curated.pkl')
    # X2 = X2['0019ef41.wav']
    # print(X)
    # print(X2)
    # print(X - X2)
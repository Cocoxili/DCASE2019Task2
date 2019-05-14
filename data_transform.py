
from util import *
from tqdm import tqdm


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
            feats = np.stack((logmel, delta, accelerate)) #(3, 64, xx)

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


def calculate_mean_and_std(X):
    X = list(X.values())
    means = []
    stds = []
    for channel in range(3):
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


if __name__ == '__main__':
    config = Config(
                    csv_train_noisy='./trn_noisy_best50s.csv',
                    # csv_train_noisy='../input/train_noisy.csv',
                    sampling_rate=44100,
                    audio_duration=1.5,
                    n_mels=128,
                    frame_weigth=100,
                    frame_shift=10,
                    batch_size=512,
                    n_folds=5,
                    features_dir="../../../features/logmel_w100_s10_m128_trim_norm",
                    model_dir='../model/mobileNetv2',
                    prediction_dir='../prediction/mobileNetv2',
                    arch='resnet50_mfcc',
                    lr=0.01,
                    pretrain=True,
                    mixup=False,
                    #  epochs=100)
                    epochs=200,
                    debug=False)

    means = np.array([-26.6642, -0.0131, -0.0028])
    stds = np.array([20.1533, 1.0592, 0.4690])

    # X_test = get_logmel_feature(subset='test')
    # save_data(os.path.join(config.features_dir, 'test.pkl'), X_test)
    #
    X_train_curated = get_logmel_feature(subset='train_curated')
    X_train_curated = logmel_normalize(X_train_curated, means, stds)
    save_data(os.path.join(config.features_dir, 'train_curated.pkl'), X_train_curated)
    #
    # X_train_noisy = get_logmel_feature(subset='train_noisy')
    # save_data(os.path.join(config.features_dir, 'train_noisy50.pkl'), X_train_noisy)

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

    # X = load_data('/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_w100_s10_m128_trim_norm/train_curated.pkl')
    # X1 = load_data('/home/cocoxili/work/freesound-audio-tagging-2019/features/logmel_w100_s10_m128_trim/test.pkl')
    # X.update(X1)
    # print(len(X))
    # calculate_mean_and_std(X)

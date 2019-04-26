
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
            X[item['fname']] = feats
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


def get_feature(subset):
    end = time.time()
    wavelist = get_wavelist(subset)
    print(len(wavelist))
    X = wave_to_logmel(subset, wavelist)
    print('Time of data transformation: ', time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - end)))
    return X


if __name__ == '__main__':
    config = Config(sampling_rate=44100,
                    audio_duration=1.5,
                    n_mels=128,
                    frame_weigth=100,
                    frame_shift=15,
                    batch_size=512,
                    n_folds=5,
                    features_dir="../features/logmel_w100_s15_m128",
                    model_dir='../model/mobileNetv2',
                    prediction_dir='../prediction/mobileNetv2',
                    arch='resnet50_mfcc',
                    lr=0.01,
                    pretrain=True,
                    mixup=False,
                    #  epochs=100)
                    epochs=200,
                    debug=False)

    X_test = get_feature(subset='test')
    save_data(os.path.join(config.features_dir, 'test.pkl'), X_test)

    X_train = get_feature(subset='train_curated')
    save_data(os.path.join(config.features_dir, 'train_curated.pkl'), X_train)

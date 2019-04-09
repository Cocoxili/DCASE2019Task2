import numpy as np
import pandas as pd
from config import Config
from util import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm


class Freesound(Dataset):
    def __init__(self, config, frame, mode, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):

        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'

        file_path = os.path.join(self.config.features_dir, filename)

        # Read and Resample the audio
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        data = data[np.newaxis, :]

        if self.mode is "train":
            # label_name = self.frame["label"][idx]
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data

    def _random_selection(self, file_path):

        input_length = self.config.audio_length
        # Read and Resample the audio
        data = load_data(file_path)

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        return data


class FreesoundLogmel(Dataset):
    def __init__(self, config, frame, mode, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'

        file_path = os.path.join(self.config.features_dir, filename)

        # Read and Resample the audio
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        # data = data[np.newaxis, :]

        if self.mode is "train":
            # label_name = self.frame["label"][idx]
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data

    def _random_selection(self, file_path):

        input_frame_length = int(self.config.audio_duration * 1000 / self.config.frame_shift)
        # Read the logmel pkl
        logmel = load_data(file_path)

        # Random offset / Padding
        if logmel.shape[2] > input_frame_length:
            max_offset = logmel.shape[2] - input_frame_length
            offset = np.random.randint(max_offset)
            data = logmel[:, :, offset:(input_frame_length + offset)]
        else:
            if input_frame_length > logmel.shape[2]:
                max_offset = input_frame_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_frame_length - logmel.shape[2] - offset)), "constant")
        return data


class FreesoundLogmelDiscontinuous(Dataset):
    def __init__(self, config, frame, mode, order=True, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.mode = mode
        self.order = order

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'

        file_path = os.path.join(self.config.features_dir, filename)

        # Read and Resample the audio
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        # data = data[np.newaxis, :]

        if self.mode is "train":
            # label_name = self.frame["label"][idx]
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data

    def _random_selection(self, file_path):

        input_frame_length = int(self.config.audio_duration * 1000 / self.config.frame_shift)
        # Read the logmel pkl
        logmel = load_data(file_path)

        # Random offset / Padding
        if logmel.shape[2] > input_frame_length:
            # print(logmel)
            # print(logmel.shape)
            index = np.random.choice(logmel.shape[2], input_frame_length)
            if self.order == True:
                index = np.sort(index)
            data = logmel[:, :, index]
            # print(data)
            # print(data.shape)
        else:
            if input_frame_length > logmel.shape[2]:
                max_offset = input_frame_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_frame_length - logmel.shape[2] - offset)), "constant")
        return data

    # def _random_selection(self, file_path):
    #
    #     input_frame_length = int(self.config.audio_duration * 1000 / self.config.frame_shift)
    #     # Read the logmel pkl
    #     logmel = load_data(file_path)
    #
    #     # Random offset / Padding
    #     if logmel.shape[2] > input_frame_length:
    #         max_offset = logmel.shape[2] - input_frame_length
    #         offset = np.random.randint(max_offset)
    #         data = logmel[:, :, offset:(input_frame_length + offset)]
    #     else:
    #         if input_frame_length > logmel.shape[2]:
    #             max_offset = input_frame_length - logmel.shape[2]
    #             offset = np.random.randint(max_offset)
    #         else:
    #             offset = 0
    #         data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_frame_length - logmel.shape[2] - offset)),
    #                       "constant")
    #     return data


class ToTensor(object):
    """
    convert ndarrays in sample to Tensors.
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)

    """
    def __call__(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data


def multilabel_to_onehot(labels, label_idx, num_class=80):
    """
    :param labels: multi-label separated by comma.
    :param num_class: number of classes, length of one-hot label.
    :return: one-hot label, such as [0, 1, 0, 0, 1,...]
    """
    # one_hot = np.zeros(num_class)
    one_hot = torch.zeros(num_class)
    for l in labels.split(','):
        one_hot[label_idx[l]] = 1.0
    return one_hot


if __name__ == "__main__":
    # config = Config(sampling_rate=44100, audio_duration=1.5, features_dir="../data-22050")
    config = Config(sampling_rate=22050,
                    audio_duration=1.5,
                    features_dir='../input/features/logmel+delta_w80_s10_m64',)
    DEBUG = True

    train_curated = pd.read_csv('../input/train_curated.csv')
    train_noisy = pd.read_csv('../input/train_noisy.csv')
    test = pd.read_csv('../input/sample_submission.csv')

    LABELS = config.labels
    # LABELS = get_classes_name()
    # ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum',
    #  'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz',
    #  'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking', 'Chink_and_clink',
    #  'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd',
    #  'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip',
    #  'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
    #  'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss',
    #  'Keys_jangling', 'Knock', 'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone',
    #  'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing',
    #  'Raindrop', 'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam',
    #  'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise',
    #  'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Writing',
    #  'Yell', 'Zipper_(clothing)']


    label_idx = {label: i for i, label in enumerate(LABELS)}
    # print(label_idx)
    train_curated.set_index("fname")
    train_noisy.set_index("fname")
    test.set_index("fname")

    train_curated["label_idx"] = train_curated['labels'].apply(multilabel_to_onehot, args=(label_idx,))

    if DEBUG:
        train_curated = train_curated[:2000]
        test = test[:2000]

    skf = KFold(n_splits=config.n_folds)

    for foldNum, (train_split, val_split) in enumerate(skf.split(train_curated)):
        print("TRAIN:", train_split, "VAL:", val_split)
        train_set = train_curated.iloc[train_split]
        train_set = train_set.reset_index(drop=True)
        val_set = train_curated.iloc[val_split]
        val_set = val_set.reset_index(drop=True)
        print(len(train_set), len(val_set))

        trainSet = FreesoundLogmel(config=config, frame=train_set,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = FreesoundLogmel(config=config, frame=val_set,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")

        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        for i, (input, target) in enumerate(train_loader):
            print(len(train_loader.dataset))
            print(i)
            # print(input)
            print(input.size())
            print(input.type())
            print(target.type())
            print(target.size())
            break

    # ---------test logmel loader------------
    # test_set = pd.read_csv('../sample_submission.csv')
    # testSet = Freesound_logmel(config=config, frame=test_set,
    #                            # transform=transforms.Compose([ToTensor()]),
    #                            mode="test")
    # # test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=1)
    # test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=1)
    # print(len(test_loader))
    # print(type(test_loader))
    # for i, input in enumerate(test_loader):
    #
    #     print(input.type())
    #     break


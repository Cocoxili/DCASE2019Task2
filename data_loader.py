from util import *


def worker_init_fn():
    seed = random.randint(0, 65535)
    np.random.seed(seed)


class FreesoundWave(Dataset):
    def __init__(self, config, frame, X, mode, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.mode = mode
        self.X = X

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        fname = self.frame["fname"][idx]

        # Read and Resample the audio
        data = self._random_selection(fname)

        data = data[np.newaxis, :]

        if self.transform is not None:
            data = self.transform(data)

        if self.mode is "train":
            label_idx = self.frame["label_idx"][idx]
            weight = self.frame['weight'][idx]
            return data, label_idx, np.float32(weight)

        elif self.mode is "test":
            return data

    def _random_selection(self, fname):

        input_length = self.config.audio_length

        data = self.X[fname]

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


class FreesoundLogmelTrain(Dataset):
    def __init__(self, config, frame, X, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        fname = self.frame["fname"][idx]
        data = self.X[fname]

        if self.transform is not None:
            data = self.transform(data)

        label_idx = self.frame["label_idx"][idx]
        weight = self.frame['weight'][idx]
        return data, label_idx, np.float32(weight)


class FreesoundLogmelVal(Dataset):
    def __init__(self, config, frame, X, transform=None, tta=1):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.X = X
        self.tta = tta

    def __len__(self):
        return self.frame.shape[0] * self.tta

    def __getitem__(self, idx):
        idx = idx // self.tta
        fname = self.frame["fname"][idx]

        data = self.X[fname]

        if self.transform is not None:
            data = self.transform(data)

        label_idx = self.frame["label_idx"][idx]
        return fname, data, label_idx


class FreesoundLogmelTest(Dataset):
    def __init__(self, config, frame, X, transform=None, tta=1):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.X = X
        self.tta = tta

    def __len__(self):
        return self.frame.shape[0] * self.tta

    def __getitem__(self, idx):
        idx = idx // self.tta
        fname = self.frame["fname"][idx]
        data = self.X[fname]
        if self.transform is not None:
            data = self.transform(data)
        return fname, data


# class FreesoundLogmel(Dataset):
#     def __init__(self, config, frame, X, mode, transform=None):
#         self.config = config
#         self.frame = frame
#         self.transform = transform
#         self.mode = mode
#         self.X = X
#
#     def __len__(self):
#         return self.frame.shape[0]
#
#     def __getitem__(self, idx):
#         fname = self.frame["fname"][idx]
#
#         # Read and Resample the audio
#         # data = self._random_selection(fname)
#
#         olddata = self.X[fname]
#         da = olddata[0, ::][np.newaxis, ::]
#         data = np.concatenate((da, da, da), axis=0)
#
#         if self.transform is not None:
#             data = self.transform(data)
#
#         if self.mode is "train":
#             label_idx = self.frame["label_idx"][idx]
#             weight = self.frame['weight'][idx]
#             return data, label_idx, np.float32(weight)
#
#         elif self.mode is "test":
#             return data


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
    def __call__(self, data):
        tensor = torch.from_numpy(data).type(torch.FloatTensor)
        return tensor


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy randomly with a given probability.

    Args:
        p (float): probability of the numpy being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return np.flip(data, axis=2).copy()
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCut2D(object):
    def __init__(self, config):
        self.config = config
        self.input_frame_length = int(self.config.audio_duration * 1000 / self.config.frame_shift)

    def __call__(self, spec):

        # Random offset / Padding
        if spec.shape[2] > self.input_frame_length:
            max_offset = spec.shape[2] - self.input_frame_length
            """
                Very strange thing: np.random.randint() will give same number every epoch
            """
            # offset = np.random.randint(max_offset)
            offset = random.randint(0, max_offset-1)
            data = spec[:, :, offset:(self.input_frame_length + offset)]
        else:
            if self.input_frame_length > spec.shape[2]:
                max_offset = self.input_frame_length - spec.shape[2]
                offset = random.randint(0, max_offset-1)
                # offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(spec,
                          ((0, 0), (0, 0), (offset, self.input_frame_length - spec.shape[2] - offset)),
                          "constant")
        return data


class RandomFrequencyMask(object):
    """
    Frequency masking is applied so that f consecutive mel frequency channels [f0, f0 + f)
    are masked, where f is first chosen from a uniform distribution from 0 to the frequency
    mask parameter max_width, and f0 is chosen from 0, ν − f). ν is the number of mel frequency
    channels.
    """
    def __init__(self, p, config, num_mask, max_width, replace='zero'):
        self.p = p
        self.config = config
        self.num_mask = num_mask
        self.max_width = max_width
        self.replace = replace

    def __call__(self, spec):
        spec_c = spec.copy()

        if random.random() < self.p:
            for i in range(self.num_mask):
                f_0 = random.randint(0, self.config.n_mels-self.max_width-1)
                f = random.randint(0, self.max_width)
                if self.replace == 'random':
                    mask = np.random.rand(spec.shape[0], f, spec.shape[2])
                elif self.replace == 'zero':
                    mask = 0
                else:
                    raise ValueError("Replace not support {} value.".format(self.replace))
                spec_c[:, f_0:f_0+f, :] = mask

        return spec_c


class RandomTimeMask(object):
    """
    Time masking is applied so that t consecutive time steps [t0, t0 + t) are masked,
    where t is first chosen from a uniform distribution from 0 to the time mask parameter
    max_width, and t0 is chosen from [0, τ − t). We introduce an upper bound on the time
    mask so that a time mask cannot be wider than p times the number of time steps.
    """
    def __init__(self, p, config, num_mask, max_width, replace='zero'):
        self.p = p
        self.config = config
        self.num_mask = num_mask
        self.max_width = max_width
        self.replace = replace

    def __call__(self, spec):
        spec_c = spec.copy()

        if random.random() < self.p:
            for i in range(self.num_mask):
                t_0 = random.randint(0, self.config.n_mels-self.max_width-1)
                t = random.randint(0, self.max_width)
                if self.replace == 'random':
                    mask = np.random.rand(spec.shape[0], spec.shape[1], t)
                elif self.replace == 'zero':
                    mask = 0
                else:
                    raise ValueError("Replace not support {} value.".format(self.replace))
                spec_c[:, :, t_0:t_0+t] = mask

        return spec_c


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    replace: 'zero', 'mean' or 'random' value
    -------------------------------------------------------------------------------------
    """

    def __init__(self, probability=1, sl=0.02, sh=0.4, r1=0.3, replace='random'):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.replace = replace

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)

                if self.replace == 'zero':
                    mask = np.zeros((3, h, w))
                elif self.replace == 'random':
                    mask = np.random.rand(3, h, w)

                img[:, x1:x1 + h, y1:y1 + w] = mask

                return img
        return img


def get_logmel_loader(foldNum, df_train_curated, df_train_noisy, X, skf, config):

    # train_split, val_split = next(
    #     itertools.islice(skf.split(np.arange(len(df_train_noisy)), df_train_noisy['labels']), foldNum, foldNum + 1))

    train_set = df_train_curated[df_train_curated['fold'] != foldNum]
    # train_set = pd.concat([train_set, df_train_noisy.iloc[train_split]], sort=True) # add noisy data
    train_set = train_set.sample(frac=1)    # shuffle
    train_set = train_set.reset_index(drop=True)

    val_set = df_train_curated[df_train_curated['fold'] == foldNum]
    # val_set = pd.concat([val_set, df_train_noisy.iloc[val_split]], sort=True)
    val_set = val_set.sample(frac=1)
    val_set = val_set.reset_index(drop=True)

    logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
                 .format(foldNum, len(train_set), len(val_set)))

    # composed = transforms.Compose([RandomHorizontalFlip(0.5)])
    composed_train = transforms.Compose([RandomCut2D(config),
                                         # RandomHorizontalFlip(0.5),
                                         RandomFrequencyMask(1, config, 1, 30),
                                         RandomTimeMask(1, config, 1, 30),
                                         # RandomErasing(),
                                         # ToTensor(),
                                        ])
    composed_val = transforms.Compose([RandomCut2D(config),
                                       # RandomFrequencyMask(1, config, 1, 30),
                                       # RandomTimeMask(1, config, 1, 30)
                                       # RandomErasing(),
                                       # ToTensor(),
                                       ])
    # define train loader and val loader
    trainSet = FreesoundLogmelTrain(config=config, frame=train_set, X=X,
                                    transform=composed_train)
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=1)

    valSet = FreesoundLogmelVal(config=config, frame=val_set, X=X,
                                transform=composed_val,
                                tta=3)
    val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def get_frame_split(foldNum, df_train_curated, df_train_noisy):

    train_set = df_train_curated[df_train_curated['fold'] != foldNum]
    # train_set = df_train_noisy[df_train_noisy['fold'] != foldNum]
    # train_set = pd.concat([train_set, df_train_noisy[df_train_noisy['fold'] != foldNum]])  # add noisy data
    train_set = train_set.sample(frac=1)    # shuffle
    train_set = train_set.reset_index(drop=True)

    val_set = df_train_curated[df_train_curated['fold'] == foldNum]
    # val_set = df_train_noisy[df_train_noisy['fold'] == foldNum]
    # val_set = pd.concat([val_set, df_train_noisy[df_train_noisy['fold'] == foldNum]])  # add noisy data
    val_set = val_set.sample(frac=1)
    val_set = val_set.reset_index(drop=True)

    logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
                 .format(foldNum, len(train_set), len(val_set)))

    return train_set, val_set


def get_wave_loader(df_train_curated, df_train_noisy, X, skf, foldNum, config):

    # Get the nth item of a generator
    train_split, val_split = next(itertools.islice(skf.split(df_train_curated), foldNum, foldNum + 1))

    train_set = df_train_curated.iloc[train_split]
    # train_set = pd.concat([train_set, df_train_noisy], sort=True)
    train_set = train_set.reset_index(drop=True)
    val_set = df_train_curated.iloc[val_split]
    val_set = val_set.reset_index(drop=True)
    logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
                 .format(foldNum, len(train_set), len(val_set)))

    composed = transforms.Compose([RandomHorizontalFlip(0.5)])

    # define train loader and val loader
    trainSet = FreesoundWave(config=config, frame=train_set, X=X,
                               transform=composed,
                               mode="train")
    # trainSet = Freesound_logmel_discontinuous(config=config, frame=train_set, order=True,
    #                      transform=transforms.Compose([ToTensor()]),
    #                      mode="train")
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

    valSet = FreesoundWave(config=config, frame=val_set, X=X,
                             transform=composed,
                             mode="train")
    # valSet = Freesound_logmel_discontinuous(config=config, frame=val_set, order=True,
    #                      transform=transforms.Compose([ToTensor()]),
    #                      mode="train")
    val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


if __name__ == "__main__":

    """
    # config = Config(sampling_rate=44100, audio_duration=1.5, features_dir="../data-22050")
    config = Config(sampling_rate=44100,
                    audio_duration=1.5,
                    features_dir='../input/features/logmel+delta_w80_s10_m64',)
    DEBUG = True

    train_curated = pd.read_csv('../input/train_curated.csv')
    train_noisy = pd.read_csv('../input/train_noisy.csv')
    test = pd.read_csv('../input/sample_submission.csv')

    LABELS = config.labels

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

    """


from util import *


def worker_init_fn():
    seed = random.randint(0, 65535)
    np.random.seed(seed)


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
        return data, label_idx


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


class FreesoundDominateMixup(Dataset):
    def __init__(self, config, alpha, curated_df, noisy_df, X, transform=None):
        self.config = config
        self.alpha = alpha
        self.curated_df = curated_df
        self.noisy_df = noisy_df
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.curated_df.shape[0]

    def __getitem__(self, idx):
        noisy_idx = random.randint(0, self.noisy_df.shape[0] - 1)
        fname_curated = self.curated_df['fname'][idx]
        fname_noisy = self.noisy_df['fname'][noisy_idx]
        # fname_curated =os.path.join(self.config.features_dir,
        #                             'train_curated',
        #                             os.path.splitext(fname_curated)[0] + '.pkl')
        fname_noisy = os.path.join(self.config.features_dir,
                                   'train_noisy_all',
                                   os.path.splitext(fname_noisy)[0] + '.pkl')

        data_curated = self.X[fname_curated]
        data_noisy = load_data(fname_noisy)

        random2d = RandomCut2D(self.config)
        data_curated = random2d(data_curated)
        data_noisy = random2d(data_noisy)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max([lam, 1 - lam])
        label_curated = self.curated_df['label_idx'][idx]
        label_noisy = self.noisy_df['label_idx'][noisy_idx]

        mixed_x = lam * data_curated + (1 - lam) * data_noisy
        mixed_y = lam * label_curated + (1 - lam) * label_noisy

        if self.transform is not None:
            mixed_x = self.transform(mixed_x)

        return mixed_x, mixed_y


class FreesoundDominateMixupWithCurated(Dataset):
    def __init__(self, config, alpha, curated_df, noisy_df, X, transform=None):
        self.config = config
        self.alpha = alpha
        self.curated_df = curated_df
        self.noisy_df = noisy_df
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.curated_df.shape[0]

    def __getitem__(self, idx):

        noisy_idx = random.randint(0, self.noisy_df.shape[0] - 1)
        fname_curated = self.curated_df['fname'][idx]
        data_curated = self.X[fname_curated]

        if random.random() < 0.2:
            fname_noisy = self.curated_df['fname'][random.randint(0, self.curated_df.shape[0] - 1)]
            data_noisy = self.X[fname_noisy]
        else:
            fname_noisy = self.noisy_df['fname'][noisy_idx]
            fname_noisy = os.path.join(self.config.features_dir,
                                       'train_noisy_all',
                                       os.path.splitext(fname_noisy)[0] + '.pkl')
            data_noisy = load_data(fname_noisy)

        random2d = RandomCut2D(self.config)
        data_curated = random2d(data_curated)
        data_noisy = random2d(data_noisy)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max([lam, 1 - lam])
        label_curated = self.curated_df['label_idx'][idx]
        label_noisy = self.noisy_df['label_idx'][noisy_idx]

        mixed_x = lam * data_curated + (1 - lam) * data_noisy
        mixed_y = lam * label_curated + (1 - lam) * label_noisy

        if self.transform is not None:
            mixed_x = self.transform(mixed_x)

        return mixed_x, mixed_y


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
    are masked, where f is first chosen from a uniform distribution from 0 to the
    max_width_ratio x ν, and f0 is chosen from [0, ν − f). ν is the number of mel
    frequency channels.
    """
    def __init__(self, p, config, num_mask, max_width_ratio, replace='zero'):
        self.p = p
        self.config = config
        self.num_mask = num_mask
        self.width = config.n_mels
        self.max_width = int(max_width_ratio * self.width)
        self.replace = replace

    def __call__(self, spec):
        spec_c = spec.copy()

        if random.random() < self.p:
            for i in range(self.num_mask):
                f_0 = random.randint(0, self.width-self.max_width-1)
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
    where t is first chosen from a uniform distribution from 0 to the
    max_width_ratio x τ, and t0 is chosen from [0, τ − t). We introduce an upper
    bound on the time mask so that a time mask cannot be wider than p times the
    number of time steps.
    """
    def __init__(self, p, config, num_mask, max_width_ratio, replace='zero'):
        self.p = p
        self.config = config
        self.num_mask = num_mask
        self.width = int(self.config.audio_duration * 1000 / self.config.frame_shift)
        self.max_width = int(max_width_ratio * self.width)
        self.replace = replace

    def __call__(self, spec):
        spec_c = spec.copy()

        if random.random() < self.p:
            for i in range(self.num_mask):
                t_0 = random.randint(0, self.width-self.max_width-1)
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


class ToTensor(object):
    def __call__(self, data):
        tensor = torch.from_numpy(data).type(torch.FloatTensor)
        return tensor


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
    val_set = val_set.reset_index(drop=True)

    logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
                 .format(foldNum, len(train_set), len(val_set)))

    return train_set, val_set


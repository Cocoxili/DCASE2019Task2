from core import *
from data_loader import *
from util import *

import torch.optim as optim
import torch.backends.cudnn as cudnn

import time

np.random.seed(1001)


def main():

    train = pd.read_csv(config.CSV_TRAIN_CURATED)

    LABELS = config.labels
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index("fname")
    train["label_idx"] = train['labels'].apply(multilabel_to_onehot, args=(label_idx,))
    train.set_index("fname")

    if DEBUG:
        train = train[:500]

    skf = KFold(n_splits=config.n_folds)

    for foldNum, (train_split, val_split) in enumerate(skf.split(train)):

        end = time.time()
        # split the dataset for cross-validation
        train_set = train.iloc[train_split]
        train_set = train_set.reset_index(drop=True)
        val_set = train.iloc[val_split]
        val_set = val_set.reset_index(drop=True)
        logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
                     .format(foldNum, len(train_set), len(val_set)))

        # define train loader and val loader
        trainSet = FreesoundLogmel(config=config, frame=train_set,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")
        # trainSet = Freesound_logmel_discontinuous(config=config, frame=train_set, order=True,
        #                      transform=transforms.Compose([ToTensor()]),
        #                      mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = FreesoundLogmel(config=config, frame=val_set,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")
        # valSet = Freesound_logmel_discontinuous(config=config, frame=val_set, order=True,
        #                      transform=transforms.Compose([ToTensor()]),
        #                      mode="train")
        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        model = run_method_by_string(config.arch)(pretrained=config.pretrain)

        if config.cuda:
            model.cuda()

        # define loss function (criterion) and optimizer
        if config.mixup:
            train_criterion = cross_entropy_onehot
        else:
            train_criterion = cross_entropy_onehot
            # train_criterion = nn.CrossEntropyLoss().cuda()

        # val_criterion = nn.CrossEntropyLoss().cuda()
        val_criterion = cross_entropy_onehot

        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)

        cudnn.benchmark = True

        train_on_fold(model, train_criterion, val_criterion,
                      optimizer, train_loader, val_loader, config, foldNum)

        time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
        logging.info("--------------Time on fold {}: {}--------------\n"
                     .format(foldNum, time_on_fold))


    # Train on the whole training set
    # foldNum = config.n_folds
    # end = time.time()
    # logging.info("Fold {0}, Train samples:{1}."
    #               .format(foldNum, len(train)))
    #
    # # define train loader and val loader
    # trainSet = Freesound_logmel(config=config, frame=train,
    #                             transform=transforms.Compose([ToTensor()]),
    #                             mode="train")
    # train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)
    #
    # model = run_method_by_string(config.arch)(pretrained=config.pretrain)
    #
    # if config.cuda:
    #      model.cuda()
    #
    # # define loss function (criterion) and optimizer
    # if config.mixup:
    #     train_criterion = cross_entropy_onehot
    # else:
    #     train_criterion = cross_entropy_onehot
    #
    # optimizer = optim.SGD(model.parameters(), lr=config.lr,
    #                       momentum=config.momentum,
    #                       weight_decay=config.weight_decay)
    #
    # cudnn.benchmark = True
    #
    # train_all_data(model, train_criterion, optimizer, train_loader, config, foldNum)
    #
    # time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - end))
    # logging.info("--------------Time on fold {}: {}--------------\n"
    #             .format(foldNum, time_on_fold))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    DEBUG = True

    config = Config(sampling_rate=22050,
                    audio_duration=1.5,
                    batch_size=128,
                    n_folds=5,
                    features_dir="../input/features/logmel+delta_w80_s10_m64",
                    model_dir='../model/logmel_delta_resnet50',
                    prediction_dir='../prediction/logmel_delta_resnet50',
                    arch='resnet50_mfcc',
                    lr=0.01,
                    pretrain=True,
                    mixup=False,
                    #  epochs=100)
                    epochs=50)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    main()

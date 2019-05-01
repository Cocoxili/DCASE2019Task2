from core import *
from data_loader import *
from util import *


def define_model():
    # model = resnet18()
    model = mobilenet_v2(num_classes=config.num_classes)
    return model
    # return Classifier(num_classes=config.num_classes)
    # return resnet50_mfcc()
    # model = run_method_by_string(config.arch)(pretrained=config.pretrain, num_classes=config.num_classes)
    # return model


def pretrain():
    end = time.time()

    vis = visdom.Visdom()

    df_train_noisy = pd.read_csv(config.CSV_TRAIN_NOISY)

    LABELS = config.labels
    label_idx = {label: i for i, label in enumerate(LABELS)}

    df_train_noisy.set_index("fname")
    df_train_noisy["label_idx"] = df_train_noisy['labels'].apply(multilabel_to_onehot, args=(label_idx,))
    df_train_noisy.set_index("fname")

    X = load_data(os.path.join(config.features_dir, 'train_noisy.pkl'))

    if config.debug:
        df_train_noisy = df_train_noisy[:200]

    train_set, val_set = train_test_split(df_train_noisy, test_size=0.2, shuffle=True, random_state=1001)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    logging.info("Pretrain: Train samples:{0}, val samples:{1}"
                 .format(len(train_set), len(val_set)))

    # define train loader and val loader
    trainSet = FreesoundLogmel(config=config, frame=train_set, X=X,
                               transform=transforms.Compose([ToTensor()]),
                               mode="train")
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

    valSet = FreesoundLogmel(config=config, frame=val_set, X=X,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")
    val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)


    model = define_model()
    # criterion = cross_entropy_onehot
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.KLDivLoss(reduction='batchmean')

    if config.cuda:
        model.cuda()
        criterion = criterion.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=config.lr,
    #                       momentum=config.momentum,
    #                       weight_decay=config.weight_decay)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           amsgrad=False)

    cudnn.benchmark = True

    lwlrap = train_on_fold(model, criterion, criterion,
                  optimizer, train_loader, val_loader, config, 0, vis)

    time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))

    logging.info("Pretrain on {}/{} noisy samples:\t lwlrp: {:.3f} \t time: {}"
                 .format(len(train_set), len(df_train_noisy), lwlrap, time_on_fold))
    #

    #
    # times = []
    # results = []
    # for foldNum in range(config.n_folds):
    #
    #     end = time.time()
    #
    #     train_loader, val_loader = get_data_loader(df_train_curated, df_train_noisy, X, skf, foldNum, config)
    #
    #     model = define_model()
    #     # criterion = cross_entropy_onehot
    #     criterion = nn.BCEWithLogitsLoss()
    #     # criterion = nn.KLDivLoss(reduction='batchmean')
    #
    #     if config.cuda:
    #         model.cuda()
    #         criterion = criterion.cuda()
    #
    #     # optimizer = optim.SGD(model.parameters(), lr=config.lr,
    #     #                       momentum=config.momentum,
    #     #                       weight_decay=config.weight_decay)
    #     optimizer = optim.Adam(params=model.parameters(),
    #                            lr=config.lr,
    #                            weight_decay=config.weight_decay,
    #                            amsgrad=False)
    #
    #     cudnn.benchmark = True
    #
    #     lwlrap = train_on_fold(model, criterion, criterion,
    #                   optimizer, train_loader, val_loader, config, foldNum, vis)
    #
    #     time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
    #     times.append(time_on_fold)
    #     results.append(lwlrap)
    #     logging.info("--------------Time on fold {}: {}--------------\n"
    #                  .format(foldNum, time_on_fold))
    #
    # for foldNum in range(config.n_folds):
    #     logging.info("Fold{}:\t lwlrp: {:.3f} \t time: {}".format(foldNum, results[foldNum], times[foldNum]))


if __name__ == "__main__":
    seed_everything(1001)

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    config = Config(
                    csv_train_noisy='./trn_noisy_best50s.csv',
                    # csv_train_noisy='../input/train_noisy.csv',
                    sampling_rate=44100,
                    audio_duration=1.5,
                    n_mels=32,
                    frame_weigth=100,
                    frame_shift=10,
                    batch_size=128,
                    n_folds=5,
                    features_dir="../features/logmel_w100_s10_m128",
                    # model_dir='../model/resnet',
                    model_dir='../model/mobileNetv2_test2',
                    # prediction_dir='../prediction/mobileNetv2_test1',
                    arch='mobileNetV2',
                    lr=1e-3,
                    eta_min=1e-5,
                    # weight_decay=5e-5,
                    pretrain=False,
                    mixup=False,
                    #  epochs=100)
                    epochs=120,
                    debug=False)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    pretrain()

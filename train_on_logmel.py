from core import *
from data_loader import *
from util import *


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def define_model():
    # model = resnet50_mfcc()
    model = mobilenet_v2(num_classes=config.num_classes)
    return model
    # return Classifier(num_classes=config.num_classes)


def main():
    df_train = pd.read_csv(config.CSV_TRAIN_CURATED)
    LABELS = config.labels
    label_idx = {label: i for i, label in enumerate(LABELS)}
    df_train.set_index("fname")
    df_train["label_idx"] = df_train['labels'].apply(multilabel_to_onehot, args=(label_idx,))
    df_train.set_index("fname")

    X = load_data(os.path.join(config.features_dir, 'train_curated.pkl'))

    if config.debug:
        df_train = df_train[:500]

    skf = KFold(n_splits=config.n_folds, shuffle=True)

    times = []
    results = []
    for foldNum in range(config.n_folds):

        end = time.time()

        train_loader, val_loader = get_data_loader(df_train, X, skf, foldNum, config)

        model = define_model()
        # criterion = cross_entropy_onehot
        criterion = nn.BCEWithLogitsLoss()

        if config.cuda:
            model.cuda()
            criterion = criterion.cuda()

        # optimizer = optim.SGD(model.parameters(), lr=config.lr,
        #                       momentum=config.momentum,
        #                       weight_decay=config.weight_decay)
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, amsgrad=False)

        cudnn.benchmark = True

        lwlrap = train_on_fold(model, criterion, criterion,
                      optimizer, train_loader, val_loader, config, foldNum)

        time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
        times.append(time_on_fold)
        results.append(lwlrap)
        logging.info("--------------Time on fold {}: {}--------------\n"
                     .format(foldNum, time_on_fold))

    for foldNum in range(config.n_folds):
        logging.info("Fold{}:\t lwlrp: {:.3f} \t time: {}".format(foldNum, results[foldNum], times[foldNum]))


if __name__ == "__main__":
    seed_everything(1001)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    config = Config(sampling_rate=44100,
                    audio_duration=1.5,
                    n_mels=128,
                    frame_weigth=100,
                    frame_shift=20,
                    batch_size=256,
                    n_folds=5,
                    features_dir="../features/logmel_w100_s20_m128",
                    model_dir='../model/mobileNetv2_test3',
                    # prediction_dir='../prediction/mobileNetv2_test1',
                    arch='resnet50_mfcc',
                    lr=3e-3,
                    pretrain=True,
                    mixup=False,
                    #  epochs=100)
                    epochs=120,
                    debug=False)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    main()

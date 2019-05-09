from core import *
from data_loader import *
from util import *


def define_model():
    model = waveMobileNet()
    return model


def train():
    vis = visdom.Visdom()

    df_train_curated = pd.read_csv(config.CSV_TRAIN_CURATED)
    df_train_noisy = pd.read_csv(config.CSV_TRAIN_NOISY)

    LABELS = config.labels
    label_idx = {label: i for i, label in enumerate(LABELS)}

    df_train_curated.set_index("fname")
    df_train_curated["label_idx"] = df_train_curated['labels'].apply(multilabel_to_onehot, args=(label_idx,))
    df_train_curated["weight"] = [1 for i in range(len(df_train_curated))]
    df_train_curated.set_index("fname")

    df_train_noisy.set_index("fname")
    df_train_noisy["label_idx"] = df_train_noisy['labels'].apply(multilabel_to_onehot, args=(label_idx,))
    df_train_noisy["weight"] = [config.noisy_weight for i in range(len(df_train_noisy))]
    df_train_noisy.set_index("fname")

    X = load_data(os.path.join(config.features_dir, 'train_curated.pkl'))
    X_nosiy = load_data(os.path.join(config.features_dir, 'train_noisy.pkl'))
    X.update(X_nosiy)

    if config.debug:
        df_train_curated = df_train_curated[:500]
        df_train_noisy = df_train_noisy[:200]

    skf = KFold(n_splits=config.n_folds, shuffle=True)

    times = []
    results = []
    for foldNum in range(config.n_folds):

        end = time.time()

        train_loader, val_loader = get_wave_loader(df_train_curated, df_train_noisy, X, skf, foldNum, config)

        model = define_model()
        # criterion = cross_entropy_onehot
        train_criterion = nn.BCEWithLogitsLoss(reduction='none')
        val_criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.KLDivLoss(reduction='batchmean')

        if config.cuda:
            model.cuda()
            # criterion = criterion.cuda()

        # optimizer = optim.SGD(model.parameters(), lr=config.lr,
        #                       momentum=config.momentum,
        #                       weight_decay=config.weight_decay)
        optimizer = optim.Adam(params=model.parameters(),
                               lr=config.lr,
                               weight_decay=config.weight_decay,
                               amsgrad=False)

        cudnn.benchmark = True

        lwlrap = train_on_fold(model, train_criterion, val_criterion,
                      optimizer, train_loader, val_loader, config, foldNum, vis)

        time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
        times.append(time_on_fold)
        results.append(lwlrap)
        logging.info("--------------Time on fold {}: {}--------------\n"
                     .format(foldNum, time_on_fold))

    for foldNum in range(config.n_folds):
        logging.info("Fold{}:\t lwlrp: {:.3f} \t time: {}".format(foldNum, results[foldNum], times[foldNum]))


if __name__ == "__main__":
    seed_everything(1001)

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    config = Config(
                    csv_train_noisy='./trn_noisy_best50s.csv',
                    # csv_train_noisy='../input/train_noisy.csv',
                    sampling_rate=44100,
                    audio_duration=1.5,
                    n_folds=5,
                    features_dir="../../../features/wave_sr44100",
                    model_dir='../model/test1',
                    # prediction_dir='../prediction/mobileNetv2_test1',
                    arch='WaveCNN',
                    lr=1e-3,
                    batch_size=64,
                    eta_min=1e-5,
                    # weight_decay=5e-6,
                    mixup=False,
                    noisy_weight=0.5,
                    epochs=120,
                    debug=False)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    train()

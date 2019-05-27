from core import *
from data_loader import *
from util import *


def define_model():
    # model = resnet18()
    model = models.mobilenet_v2(num_classes=config.num_classes, pretrained=config.pretrain)
    # model = mobilenetv2(pretrained=config.pretrain)
    # model = models.shufflenetv2_x0_5(num_classes=config.num_classes)
    # model = models.shufflenetv2_x1_0(num_classes=config.num_classes)
    # model = models.shufflenetv2_x1_5(num_classes=config.num_classes)
    # model = models.shufflenetv2_x2_0(num_classes=config.num_classes)
    # checkpoint = '../model/mobileNetv2_test1/model_best.0.pth.tar'
    # print("=> loading checkpoint '{}'".format(checkpoint))
    # checkpoint = torch.load(checkpoint)
    # best_lwlrap = checkpoint['best_lwlrap']
    # model.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint, best_lwlrap: {:.2f}".format(best_lwlrap))
    # return model
    # return Classifier(num_classes=config.num_classes)
    # return resnet50(config.pretrain)
    # return densenet121(num_classes=config.num_classes)
    # model = run_method_by_string(config.arch)(pretrained=config.pretrain, num_classes=config.num_classes)
    return model
    # return vgg11(num_classes=config.num_classes)
    # return Baseline()
    # return TestCNN3()
    # return pretrainedmodels.models.dpn98(num_classes=80, pretrained=False)
    # return pretrainedmodels.resnext101_32x4d(num_classes=80, pretrained=None)


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
    # X_nosiy = load_data(os.path.join(config.features_dir, 'train_noisy50.pkl'))
    # X.update(X_nosiy)

    if config.debug:
        df_train_curated = df_train_curated[:500]
        df_train_noisy = df_train_noisy[:200]

    times = []
    results = []
    for foldNum in range(config.n_folds):

        end = time.time()

        # train_loader, val_loader = get_logmel_loader(foldNum, df_train_curated, df_train_noisy, X, skf, config)
        train_set, val_set = get_frame_split(foldNum, df_train_curated, df_train_noisy)

        ckp = os.path.join('../model/test3', 'model_best.%d.pth.tar' % foldNum)
        checkpoint = torch.load(ckp)
        model = define_model()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint {}, best_lwlrap: {:.4f} @ {}"
              .format(ckp, checkpoint['best_lwlrap'], checkpoint['epoch']))

        # criterion = cross_entropy_onehot
        train_criterion = nn.BCEWithLogitsLoss(reduction='none')
        # train_criterion = SoftBCE
        # train_criterion = Q_BCE

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
                               optimizer, train_set, val_set, X, config, foldNum, vis)

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

    config = Config(csv_train_curated='train_curated_stratified.csv',
                    csv_train_noisy='./noisy50_stratified.csv',
                    # csv_train_noisy='../input/train_noisy.csv',
                    sampling_rate=44100,
                    audio_duration=1.5,
                    frame_weigth=100,
                    frame_shift=5,
                    n_folds=5,
                    features_dir="../../../features/logmel_w100_s5_m128_trim_norm",
                    # model_dir='../model/resnet',
                    model_dir='../model/test1',
                    # prediction_dir='../prediction/mobileNetv2_test1',
                    arch='mobilenet',
                    batch_size=32,
                    lr=1e-4,
                    eta_min=5e-6,
                    weight_decay=5e-6,
                    mixup=False,
                    noisy_weight=1,
                    early_stopping=True,
                    label_smoothing=False,
                    epochs=130,
                    pretrain=False,
                    debug=False)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    train()

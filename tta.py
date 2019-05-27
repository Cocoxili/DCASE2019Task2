
from util import *
from data_loader import *


def define_model():
    # model = resnet18()
    model = models.mobilenet_v2(num_classes=config.num_classes)
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


def validate(duration_set, num_clips):

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

    for foldNum in range(config.n_folds):
        val_set = df_train_curated[df_train_curated['fold'] == foldNum]
        val_set = val_set.reset_index(drop=True)
        print("Prediction on Fold {0}, Val samples:{1}".format(foldNum, len(val_set)))

        ckp = os.path.join(config.model_dir, 'model_best.%d.pth.tar' % foldNum)
        checkpoint = torch.load(ckp)
        model = define_model()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint {}, best_lwlrap: {:.4f} @ {}"
              .format(ckp, checkpoint['best_lwlrap'], checkpoint['epoch']))
        if config.cuda is True:
            model.cuda()
        model.eval()

        prediction = []
        for audio_duration in duration_set:
            print('audio duration: {}'.format(audio_duration))
            val_preds, labels = val_on_fold(model, val_set, X, audio_duration, num_clips)
            # val_preds = rank_array(val_preds)
            prediction.append(val_preds)

        prediction = np.array(prediction)

        prediction = prediction.sum(axis=0)
        # prediction = prediction.max(axis=0)

        per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(labels, prediction)
        lwlrap = np.sum(per_class_lwlrap * weight_per_class)
        print('overall lwlrap on fold {}: {:.3f}'.format(foldNum, lwlrap))


def val_on_fold(model, val_set, X, audio_duration, num_clips):

    end = time.time()

    config.audio_duration = audio_duration
    composed_val = transforms.Compose([RandomCut2D(config),
                                       # RandomFrequencyMask(1, config, 1, 30),
                                       # RandomTimeMask(1, config, 1, 30)
                                       # RandomErasing(),
                                       # ToTensor(),
                                       ])
    valSet = FreesoundLogmelVal(config=config, frame=val_set, X=X,
                                transform=composed_val,
                                tta=num_clips)

    val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=1)

    all_fname = []
    all_pred = []
    all_label = []

    with torch.no_grad():
        for i, (fname, input, target) in enumerate(val_loader):
            if config.cuda:
                input = input.cuda()

            # compute output
            output = model(input)
            all_fname.extend(fname)
            all_pred.append(output.cpu().numpy())
            all_label.append(target.numpy())

    val_preds = pd.DataFrame(data=np.concatenate(all_pred),
                              index=all_fname,
                              columns=map(str, range(config.num_classes)))
    val_preds = val_preds.groupby(level=0).sum()
    # val_preds = val_preds.groupby(level=0).max()

    val_labels = pd.DataFrame(data=np.concatenate(all_label),
                              index=all_fname,
                              columns=map(str, range(config.num_classes)))
    val_labels = val_labels.groupby(level=0).first()

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(val_labels.values, val_preds.values)
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)
    #
    # measure elapsed time
    elapse = time.time() - end

    print('Test. Time {test_time:.1f} *lwlrap {lwlrap:.3f}*'
                 .format(test_time=elapse, lwlrap=lwlrap))

    # calculate lwlrap on each class
    # lwlrap_of_class = pd.DataFrame(columns=['fname', 'lwlrap'])
    # lwlrap_of_class['fname'] = config.labels
    # lwlrap_of_class['lwlrap'] = per_class_lwlrap
    # lwlrap_of_class = lwlrap_of_class.sort_values(by='lwlrap')
    # logging.info('{}'.format(lwlrap_of_class))

    return val_preds.values, val_labels.values


def test(duration_set, num_clips):
    """
    Test with augmentation
    :param num_clips: clips number of each duration view.
    :return: Prediction on n_fold models.
    """
    test_df = pd.read_csv(config.CSV_SBM)
    #  test_set = test_set[:50] # for debug

    test_df.set_index("fname")
    X = load_data(os.path.join(config.features_dir, 'test.pkl'))

    predictions = []
    for foldNum in range(config.n_folds):
        end = time.time()

        print("Prediction on Fold {0}, Val samples:{1}".format(foldNum, len(test_df)))
        ckp = os.path.join(config.model_dir, 'model_best.%d.pth.tar' % foldNum)
        checkpoint = torch.load(ckp)
        model = define_model()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint {}, best_lwlrap: {:.4f} @ {}"
              .format(ckp, checkpoint['best_lwlrap'], checkpoint['epoch']))
        if config.cuda is True:
            model.cuda()
        model.eval()

        for audio_duration in duration_set:
            print('audio duration: {}'.format(audio_duration))
            fn, pred = test_on_fold(model, test_df, X, audio_duration, num_clips)
            # print(pred)
            # pred = rank_array(pred)
            # pred = torch.from_numpy(pred)
            # pred = torch.softmax(pred, dim=1)
            # print(pred)
            predictions.append(pred)
        # measure elapsed time
        print('Time {:.1f}'.format(time.time()-end))

    predictions = np.array(predictions)
    print(predictions.shape)
    predictions = predictions.mean(axis=0)
    print(predictions.shape)
    save_to_csv(fn, predictions, 'test_predictions.csv')


def test_ensamble(duration_set, num_clips):
    """
    Test with augmentation
    :param num_clips: clips number of each duration view.
    :return: Prediction on n_fold models.
    """
    test_df = pd.read_csv(config.CSV_SBM)
    #  test_set = test_set[:50] # for debug

    test_df.set_index("fname")
    X = load_data(os.path.join(config.features_dir, 'test.pkl'))

    predictions = []
    for foldNum in range(config.n_folds):
        end = time.time()

        print("Prediction on Fold {0}, Val samples:{1}".format(foldNum, len(test_df)))
        ckp = os.path.join(config.model_dir, 'model_best.%d.pth.tar' % foldNum)
        checkpoint = torch.load(ckp)
        model = define_model()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint {}, best_lwlrap: {:.4f} @ {}"
              .format(ckp, checkpoint['best_lwlrap'], checkpoint['epoch']))
        if config.cuda is True:
            model.cuda()
        model.eval()

        for audio_duration in duration_set:
            print('audio duration: {}'.format(audio_duration))
            fn, pred = test_on_fold(model, test_df, X, audio_duration, num_clips)
            # print(pred)
            # pred = rank_array(pred)
            # pred = torch.from_numpy(pred)
            # pred = torch.softmax(pred, dim=1)
            # print(pred)
            predictions.append(pred)

        # model 2
        print("Prediction on Fold {0}, Val samples:{1}".format(foldNum, len(test_df)))
        ckp = os.path.join('../model/test2', 'model_best.%d.pth.tar' % foldNum)
        checkpoint = torch.load(ckp)
        model = define_model()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint {}, best_lwlrap: {:.4f} @ {}"
              .format(ckp, checkpoint['best_lwlrap'], checkpoint['epoch']))
        if config.cuda is True:
            model.cuda()
        model.eval()

        for audio_duration in duration_set:
            print('audio duration: {}'.format(audio_duration))
            fn, pred = test_on_fold(model, test_df, X, audio_duration, num_clips)
            # print(pred)
            # pred = rank_array(pred)
            # pred = torch.from_numpy(pred)
            # pred = torch.softmax(pred, dim=1)
            # print(pred)
            predictions.append(pred)
        # measure elapsed time
        print('Time {:.1f}'.format(time.time()-end))

    predictions = np.array(predictions)
    print(predictions.shape)
    predictions = predictions.mean(axis=0)
    print(predictions.shape)
    save_to_csv(fn, predictions, 'test_predictions.csv')


def test_on_fold(model, test_df, X, audio_duration, num_clips):

    config.audio_duration = audio_duration
    composed_val = transforms.Compose([RandomCut2D(config),
                                       # RandomFrequencyMask(1, config, 1, 30),
                                       # RandomTimeMask(1, config, 1, 30)
                                       # RandomErasing(),
                                       # ToTensor(),
                                       ])
    testSet = FreesoundLogmelTest(config=config, frame=test_df, X=X,
                                transform=composed_val,
                                tta=num_clips)

    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

    all_fname = []
    all_pred = []

    with torch.no_grad():
        for i, (fname, input) in enumerate(test_loader):
            if config.cuda:
                input = input.cuda()

            # compute output
            output = model(input)

            all_fname.extend(fname)
            all_pred.append(output.cpu().numpy())

    test_preds = pd.DataFrame(data=np.concatenate(all_pred),
                              index=all_fname,
                              columns=map(str, range(config.num_classes)))
    test_preds = test_preds.groupby(level=0).sum()

    return test_preds.index.values, test_preds.values


def save_to_csv(files_name, prediction, file):
    df = pd.DataFrame(index=files_name, data=prediction)
    df.index.name = 'fname'
    df.columns = config.labels
    path = os.path.join(config.prediction_dir, file)
    df.to_csv(path)


if __name__ == "__main__":
    seed_everything(1001)

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    config = Config(csv_train_curated='train_curated_stratified.csv',
                    csv_train_noisy='./trn_noisy_best50s.csv',
                    # csv_train_noisy='../input/train_noisy.csv',

                    sampling_rate=44100,
                    audio_duration=1.5,
                    frame_weigth=100,
                    frame_shift=5,

                    features_dir="../../../features/logmel_w100_s5_m128_trim_norm",
                    model_dir='../model/test1',
                    prediction_dir='../prediction/test4',

                    # arch='MobileNetV2',
                    mixup=False,
                    noisy_weight=1,
                    early_stopping=True,
                    debug=False)

    # make_cv_prediction()
    # make_test_prediction()
    # validate(duration_set=[4], num_clips=10)
    # validate(duration_set=[4], num_clips=10)
    # validate(duration_set=[2, 3, 4, 5], num_clips=3)
    # test(duration_set=[3, 4, 5], num_clips=5)
    # test(duration_set=[5], num_clips=10)
    test_ensamble(duration_set=[3, 4, 5], num_clips=5)

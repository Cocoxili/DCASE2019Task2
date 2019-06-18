
from util import *
from data_loader import *


def define_model():
    # model = resnet18()
    # model = models.mobilenet_v2(num_classes=config.num_classes)
    # model = mobilenetv2(pretrain=config.pretrain, num_classes=config.num_classes, width_mult=1.5)
    # model = mobilenetv3(pretrain=config.pretrain, n_class=80, mode='large', width_mult=1.5)
    model = mobilenetv2(pretrained=config.pretrain)
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


def predict_on_fold(model, noisy_set, audio_duration, num_clips):

    config.audio_duration = audio_duration
    composed_val = transforms.Compose([RandomCut2D(config),
                                       # RandomFrequencyMask(1, config, 1, 30),
                                       # RandomTimeMask(1, config, 1, 30)
                                       # RandomErasing(),
                                       # ToTensor(),
                                       ])
    noisySet = FreesoundNoisy(config=config, noisy_df=noisy_set,
                              transform=composed_val, tta=num_clips)

    noisy_loader = DataLoader(noisySet, batch_size=config.batch_size, shuffle=False, num_workers=1)

    all_fname = []
    all_pred = torch.zeros((1, config.num_classes)).cuda()
    all_label = torch.zeros((1, config.num_classes)).cuda()

    with torch.no_grad():
        for i, (fname, input, target) in enumerate(noisy_loader):
            if config.cuda:
                input = input.cuda()
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            all_fname.extend(fname)
            all_pred = torch.cat((all_pred, output), dim=0)
            all_label = torch.cat((all_label, target), dim=0)

    all_pred = all_pred[1:].cpu().numpy()
    all_label = all_label[1:].cpu().numpy()
    #         all_pred.append(output.cpu().numpy())
    #         all_label.append(target.numpy())
    #

    noisy_preds = pd.DataFrame(data=all_pred,
                              index=all_fname,
                              columns=map(str, range(config.num_classes)))
    noisy_preds = noisy_preds.groupby(level=0).sum()

    noisy_labels = pd.DataFrame(data=all_label,
                              index=all_fname,
                              columns=map(str, range(config.num_classes)))
    noisy_labels = noisy_labels.groupby(level=0).first()

    return noisy_preds.index.values, noisy_preds.values, noisy_labels.values


def noisy_predict(duration_set, num_clips):
    LABELS = config.labels
    label_idx = {label: i for i, label in enumerate(LABELS)}

    df_train_noisy = pd.read_csv(config.CSV_TRAIN_NOISY)
    df_train_noisy = df_train_noisy.sort_values(by='fname')
    df_train_noisy["label_idx"] = df_train_noisy['labels'].apply(multilabel_to_onehot, args=(label_idx,))

    for foldNum in range(config.n_folds):
    # for foldNum in range(1):
        print("Prediction on Fold {0}, Val samples:{1}".format(foldNum, len(df_train_noisy)))

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
            fn, noisy_preds, labels = predict_on_fold(model, df_train_noisy, audio_duration, num_clips)
            prediction.append(noisy_preds)

    prediction = np.array(prediction)
    prediction = prediction.mean(axis=0)
    save_data('preds.pkl', prediction)
    prediction = top_array(prediction, th=78)
    print(prediction.shape)

    hits = (prediction * labels).sum(axis=1)

    df_train_noisy['hits'] = hits
    df_preds = df_train_noisy[df_train_noisy['hits'] > 0]
    df_preds = df_preds.drop('label_idx', axis=1)

    df_preds.to_csv('filtered_noisy.csv', index=False)
    print(df_preds)
    return df_preds


def top_array(a, th=78):
    """
    Re-value each row of a 2d array, span from 0 to len(a)
    example:
    a = [0.52, 0.45, 0.71, 0.62],
        [0.51, 0.56, 0.02, 0.65]]
    rank = [[0., 0., 1., 0.],
            [0., 0., 0., 1.]]
    """
    rank = np.zeros_like(a)
    for row in range(a.shape[0]):
        for i, p in enumerate(a.argsort()[row]):
            if i > th:
                rank[row][p] = 1
    return rank


def save_to_csv(files_name, prediction, file):
    df = pd.DataFrame(index=files_name, data=prediction)
    df.index.name = 'fname'
    df.columns = config.labels
    path = os.path.join(config.prediction_dir, file)
    df.to_csv(path)


if __name__ == "__main__":
    seed_everything(1001)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    config = Config(csv_train_curated='train_curated_stratified.csv',
                    csv_train_noisy='../../../input/train_noisy.csv',
                    # csv_train_noisy='../input/train_noisy.csv',

                    audio_duration=1.5,
                    frame_weigth=100,
                    frame_shift=5,

                    features_dir="../../../features/logmel_w100_s5_m128_trim_norm",
                    model_dir='../model/DominateMixup_yv2_cv862',
                    prediction_dir='../prediction/test1',

                    # arch='MobileNetV2',
                    pretrain=None,
                    debug=False)

    # make_cv_prediction()
    # make_test_prediction()
    # validate(duration_set=[4], num_clips=10)
    # validate(duration_set=[4], num_clips=10)
    # validate(duration_set=[2, 3, 4, 5], num_clips=3)
    # test(duration_set=[3, 4, 5], num_clips=5)
    # test(duration_set=[5], num_clips=10)
    # test_ensamble(duration_set=[3, 4, 5], num_clips=5)
    noisy_predict(duration_set=[3, 4, 5], num_clips=5)
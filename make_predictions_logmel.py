
from util import *


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


def make_cv_prediction():

    df_train_curated = pd.read_csv(config.CSV_TRAIN_CURATED)
    # train = train[:100] # for debug

    LABELS = config.labels
    label_idx = {label: i for i, label in enumerate(LABELS)}

    df_train_curated.set_index("fname")
    df_train_curated["label_idx"] = df_train_curated['labels'].apply(multilabel_to_onehot, args=(label_idx,))
    df_train_curated.set_index("fname")

    X = load_data(os.path.join(config.features_dir, 'train_curated.pkl'))
    predictions = np.zeros((1, config.num_classes))
    file_names = []
    for foldNum in range(config.n_folds):

        val_set = df_train_curated[df_train_curated['fold'] == foldNum]
        val_set = val_set.reset_index(drop=True)

        print("Prediction on Fold {0}, Val samples:{1}".format(foldNum, len(val_set)))

        ckp = os.path.join(config.model_dir, 'model_best.%d.pth.tar' % foldNum)
        fn, pred = predict_one_model_with_logmel(ckp, val_set, X)

        file_names.extend(fn)
        predictions = np.concatenate((predictions, pred.cpu().numpy()))

    predictions = predictions[1:]
    save_to_csv(file_names, predictions, 'oof_predictions.csv')


def make_test_prediction(mean_method='arithmetic'):

    test_df = pd.read_csv(config.CSV_SBM)
    #  test_set = test_set[:50] # for debug

    test_df.set_index("fname")
    X = load_data(os.path.join(config.features_dir, 'test.pkl'))

    pred_list = []
    for foldNum in range(config.n_folds):
        ckp = os.path.join(config.model_dir, 'model_best.%d.pth.tar' % foldNum)
        fn, pred = predict_one_model_with_logmel(ckp, test_df, X)

        pred = pred.cpu().numpy()
        pred_list.append(pred)

    if mean_method == 'arithmetic':
        predictions = np.zeros_like(pred_list[0])
        for pred in pred_list:
            predictions = predictions + pred
        predictions = predictions / len(pred_list)
        print(predictions.shape)
    elif mean_method == 'geometric':
        predictions = np.ones_like(pred_list[0])
        for pred in pred_list:
            predictions = predictions * pred
        predictions = predictions ** (1. / len(pred_list))
    else:
        raise ValueError("mean_method not support {} value.".format(mean_method))

    save_to_csv(fn, predictions, 'test_predictions.csv')


def predict_one_model_with_logmel(checkpoint, frame, X):
    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    #  model = checkpoint['model']
    model = define_model()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    print("=> loaded checkpoint, best_lwlrap: {:.4f} @ {}"
          .format(checkpoint['best_lwlrap'], checkpoint['epoch']))

    input_frame_length = int(config.audio_duration * 1000 / config.frame_shift)
    stride = 20

    if config.cuda is True:
        model.cuda()

    model.eval()

    prediction = torch.zeros((1, config.num_classes)).cuda()
    file_names = []
    with torch.no_grad():

        for idx in tqdm(range(frame.shape[0])):

            logmel = X[frame['fname'][idx]]
            if logmel.shape[2] < input_frame_length:
                logmel = np.pad(logmel, ((0, 0), (0, 0), (0, input_frame_length - logmel.shape[2])), "constant")

            wins_data = []
            for j in range(0, logmel.shape[2] - input_frame_length + 1, stride):
                win_data = logmel[:, :, j: j + input_frame_length]
                wins_data.append(win_data)

            wins_data = np.array(wins_data)

            data = torch.from_numpy(wins_data).type(torch.FloatTensor)

            if config.cuda:
                data = data.cuda()

            output = model(data)
            output = torch.sum(output, dim=0, keepdim=True)
            output = F.sigmoid(output)

            prediction = torch.cat((prediction, output), dim=0)
            file_names.append(frame["fname"][idx])

    prediction = prediction[1:]
    return file_names, prediction


def save_to_csv(files_name, prediction, file):
    df = pd.DataFrame(index=files_name, data=prediction)
    df.index.name = 'fname'
    df.columns = config.labels
    path = os.path.join(config.prediction_dir, file)
    df.to_csv(path)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    config = Config(csv_train_curated='train_curated_stratified.csv',
                    csv_train_noisy='./trn_noisy_best50s.csv',
                    # csv_train_noisy='../input/train_noisy.csv',

                    sampling_rate=44100,
                    audio_duration=1.5,
                    frame_weigth=100,
                    frame_shift=10,

                    features_dir="../../../features/logmel_w100_s10_m128_trim_norm",
                    model_dir='../model/test1',
                    prediction_dir='../prediction/test1',

                    # arch='MobileNetV2',
                    mixup=False,
                    noisy_weight=1,
                    early_stopping=True,
                    debug=False)

    make_cv_prediction()
    make_test_prediction()

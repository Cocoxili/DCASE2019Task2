"""
set parameters by instantiating Config

"""


class Config(object):
    def __init__(self,
                 csv_train_curated='../../../input/train_curated.csv',
                 csv_train_noisy='../../../input/train_noisy.csv',
                 csv_sbm='../../../input/sample_submission.csv',
                 sampling_rate=44100, audio_duration=1.5,
                 train_curated_dir='../../../input/train_curated',
                 train_noisy_dir='../../../input/train_noisy',
                 test_dir='../../../input/test',
                 features_dir='../../../features/logmel+delta_w80_s10_m64',
                 model_dir='../model',
                 prediction_dir='../prediction',
                 arch='resnet50', pretrain=False,
                 cuda=True, print_freq=100, epochs=80,
                 batch_size=32,
                 momentum=0.9, weight_decay=0,
                 n_folds=5, lr=0.01, eta_min=1e-5,
                 n_mels=128, frame_weigth=100, frame_shift=10,
                 noisy_weight=0.5,
                 early_stopping=False,
                 label_smoothing=False,
                 mixup=False,
                 debug=False):

        self.CSV_TRAIN_CURATED = csv_train_curated
        self.CSV_TRAIN_NOISY = csv_train_noisy
        self.CSV_SBM = csv_sbm

        self.labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum',
         'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz',
         'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking', 'Chink_and_clink',
         'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd',
         'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip',
         'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
         'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss',
         'Keys_jangling', 'Knock', 'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone',
         'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing',
         'Raindrop', 'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard',
         'Slam', 'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
         'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet',
         'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']
        self.num_classes = len(self.labels)

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.audio_length = int(self.sampling_rate * self.audio_duration)
        self.train_curated_dir = train_curated_dir
        self.train_noisy_dir = train_noisy_dir
        self.test_dir = test_dir
        self.features_dir = features_dir
        self.model_dir = model_dir
        self.prediction_dir = prediction_dir
        self.arch = arch
        self.pretrain = pretrain
        self.cuda = cuda
        self.print_freq = print_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_folds = n_folds
        self.lr = lr
        self.eta_min = eta_min

        self.n_fft = int(frame_weigth / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.frame_weigth = frame_weigth
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)

        self.noisy_weight = noisy_weight
        self.early_stopping = early_stopping
        self.label_smoothing = label_smoothing
        self.mixup = mixup
        self.debug = debug


if __name__ == "__main__":
    config = Config()

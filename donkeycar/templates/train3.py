#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow. Uses the data written by the
donkey v2.2 tub writer,but faster training with proper sampling of distribution
over tubs.
Has settings for continuous training that will look for new files as it trains.
Modify on_best_model if you wish continuous training to update your pi as it
builds. You can drop this in your ~/mycar dir.
Basic usage should feel familiar: python train.py --model models/mypilot


Usage:
    train.py [convert|remote]
    [--tub=<tub1,tub2,..tubn>]
    [--exclude=<pattern1,pattern2>]
    [--file=<file> ...]
    [--model=<model>]
    [--transfer=<model>]
    [--type=(linear|latent|categorical|rnn|imu|behavior|3d|look_ahead|tensorrt_linear|tflite_linear|coral_tflite_linear)]
    [--figure_format=<figure_format>]
    [--nn_size=<nn_size>]
    [--continuous] [--aug] [--frac=<training fraction>] [--dry]

Options:
    -h --help              Show this screen.
    -f --file=<file>       A text file containing paths to tub files, one per line. Option may be used more than once.
    --figure_format=png    The file format of the generated figure (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html), e.g. 'png', 'pdf', 'svg', ...
"""

import json
import zlib
from os.path import basename, join, splitext, dirname
import pickle
import datetime
from tqdm import tqdm
from tensorflow.python import keras
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.callbacks import TensorBoard
from docopt import docopt

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler, Tub
from donkeycar.parts.keras import KerasIMU, KerasCategorical, KerasBehavioral, \
    KerasLatent, KerasLocalizer, KerasSquarePlusImu, KerasWorldImu, WorldMemory
from donkeycar.parts.tflite import keras_model_to_tflite
from donkeycar.parts.augment import augment_image
from donkeycar.utils import *

figure_format = 'png'
DIV_ONE_BYTE = 1.0 / 255.0

'''
matplotlib can be a pain to setup on a Mac. So handle the case where it is
absent. When present, use it to generate a plot of training results.
'''
try:
    import matplotlib.pyplot as plt
    do_plot = True
except ImportError:
    do_plot = False
    print("matplotlib not installed")


'''
Tub management
'''


def make_key(sample):
    tub_path = sample['tub_path']
    index = sample['index']
    return tub_path + str(index)


def make_next_key(sample, index_offset):
    tub_path = sample['tub_path']
    index = sample['index'] + index_offset
    return tub_path + str(index)


def collate_records(records, gen_records, cfg):
    '''
    open all the .json records from records list passed in, read their contents,
    add them to a list of gen_records, passed in. use the opts dict to
    specify config choices
    '''
    print('Collating %d records ...' % (len(records)))
    use_speed = getattr(cfg, 'USE_SPEED_FOR_MODEL', True)
    throttle_key = 'car/speed' if use_speed else 'user/throttle'
    throttle_mult = 1.0 / cfg.MAX_SPEED
    accel_mult = 1.0 / cfg.IMU_ACCEL_NORM
    gyro_mult = 1.0 / cfg.IMU_GYRO_NORM

    print('Collating records, using', throttle_key, 'for training:')
    for record_path in tqdm(records):

        base_path = os.path.dirname(record_path)
        index = get_record_index(record_path)
        sample = dict(index=index, tub_path=base_path, record_path=record_path)
        key = make_key(sample)
        if key in gen_records:
            continue
        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except:
            continue

        sample["json_data"] = json_data
        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(base_path, image_filename)
        sample["image_path"] = image_path

        # use pilot angle if present and non-null (in json null <-> None),
        # meaning the tub was driven by auto-pilot
        is_ai = 'pilot/angle' in json_data and json_data['pilot/angle']
        angle = float(json_data['pilot/angle'] if is_ai
                      else json_data['user/angle'])
        # normalising throttle if it is speed
        throttle = float(json_data[throttle_key])
        if use_speed:
            throttle = min(1.0, throttle_mult * throttle)

        sample['angle'] = angle
        sample['throttle'] = throttle
        sample['img_data'] = json_data.get('encoder/image_latent')

        if 'car/accel' in json_data:
            accel = clamp_and_norm(json_data['car/accel'], accel_mult)
            gyro = clamp_and_norm(json_data['car/gyro'], gyro_mult)
            sample['imu_array'] = np.concatenate((accel, gyro))

        # Initialise 'train' to False
        sample['train'] = False
        gen_records[key] = sample


def save_json_and_weights(model, filename):
    '''
    given a keras model and a .h5 filename, save the model file
    in the json format and the weights file in the h5 format
    '''
    if not '.h5' == filename[-3:]:
        raise Exception("Model filename should end with .h5")

    arch = model.to_json()
    json_fnm = filename[:-2] + "json"
    weights_fnm = filename[:-2] + "weights"

    with open(json_fnm, "w") as outfile:
        parsed = json.loads(arch)
        arch_pretty = json.dumps(parsed, indent=4, sort_keys=True)
        outfile.write(arch_pretty)

    model.save_weights(weights_fnm)
    return json_fnm, weights_fnm


class MyCPCallback(keras.callbacks.ModelCheckpoint):
    '''
    custom callback to interact with best val loss during continuous training
    '''

    def __init__(self, cfg=None, pilot_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pilot_data = pilot_data
        self.cfg = cfg

    def on_epoch_end(self, epoch, logs=None):
        current_best = self.best
        super().on_epoch_end(epoch, logs)
        # if val_loss improved we save the pilot data base
        if self.best != current_best and self.pilot_data:
            self.pilot_data['Accuracy'] = self.best
            self.pilot_data['Date'] \
                = datetime.datetime.now().isoformat(sep=' ', timespec='minutes')
            # save model data
            with open(self.filepath.replace('.h5', '.json'), 'w') as f:
                json.dump(self.pilot_data, f)


def generator(kl, data, cfg, is_train_set=True):
    batch_size = cfg.BATCH_SIZE
    has_imu = type(kl) in [KerasIMU, KerasSquarePlusImu, KerasWorldImu]
    has_bvh = type(kl) is KerasBehavioral
    img_out = type(kl) is KerasLatent
    loc_out = type(kl) is KerasLocalizer
    imu_dim = getattr(cfg, 'IMU_DIM', 6)

    while True:
        batch_data = []
        keys = list(data.keys())
        random.shuffle(keys)

        if type(kl.model.output) is list:
            model_out_shape = (2, 1)
        else:
            model_out_shape = kl.model.output.shape

        if img_out:
            import cv2

        for key in keys:
            if not key in data:
                continue
            record = data[key]
            if record['train'] != is_train_set:
                continue
            batch_data.append(record)

            if len(batch_data) == batch_size:
                inputs_img = []
                inputs_imu = []
                inputs_bvh = []
                angles = []
                throttles = []
                out_img = []
                out_loc = []
                out = []

                for record in batch_data:
                    # get image data if we don't already have it
                    if record['img_data'] is None:
                        filename = record['image_path']
                        img_arr = load_scaled_image_arr(filename, cfg)
                        if img_arr is None:
                            break
                        if aug:
                            img_arr = augment_image(img_arr)
                        if cfg.CACHE_IMAGES:
                            record['img_data'] = img_arr
                    else:
                        img_arr = record['img_data']

                    if img_out:
                        rz_img_arr = cv2.resize(img_arr, (127, 127)) \
                                     * DIV_ONE_BYTE
                        out_img.append(rz_img_arr[:, :, 0]
                                       .reshape((127, 127, 1)))
                    if loc_out:
                        out_loc.append(record['location'])
                    if has_imu:
                        inputs_imu.append(record['imu_array'][:imu_dim])
                    if has_bvh:
                        inputs_bvh.append(record['behavior_arr'])

                    inputs_img.append(img_arr)
                    angles.append(record['angle'])
                    throttles.append(record['throttle'])
                    out.append([record['angle'], record['throttle']])

                if img_arr is None:
                    continue

                shape = (batch_size, cfg.TARGET_H, cfg.TARGET_W, cfg.TARGET_D)
                img_arr = np.array(inputs_img).reshape(shape)

                if has_imu:
                    X = [img_arr, np.array(inputs_imu)]
                elif has_bvh:
                    X = [img_arr, np.array(inputs_bvh)]
                else:
                    X = [img_arr]

                if img_out:
                    y = [out_img, np.array(angles), np.array(throttles)]
                elif out_loc:
                    y = [np.array(angles), np.array(throttles),
                         np.array(out_loc)]
                elif model_out_shape[1] == 2:
                    y = [np.array([out]).reshape(batch_size, 2)]
                else:
                    y = [np.array(angles), np.array(throttles)]

                yield X, y

                batch_data = []


def train(cfg, tub_names, model_name, transfer_model,
          model_type, aug, exclude=None, train_frac=None, dry=False):
    """
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    """
    verbose = cfg.VERBOSE_TRAIN
    pilot_num = 0
    pilot_data = {}
    if model_name is None:
        model_name, pilot_num = auto_generate_model_name()
        model_name += ".h5"
    pilot_data['Num'] = pilot_num

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE

    if model_name and not '.h5' == model_name[-3:]:
        raise Exception("Model filename should end with .h5")

    if aug:
        print("Using data augmentation")

    kl = get_model_by_type(model_type, cfg=cfg)
    print('Training with model type', kl.model_id())
    tub_names = handle_transfer(cfg, kl, pilot_data, transfer_model, tub_names)

    if cfg.OPTIMIZER:
        kl.set_optimizer(cfg.OPTIMIZER, cfg.LEARNING_RATE,
                         cfg.LEARNING_RATE_DECAY)
    kl.compile()
    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    print('Training new pilot:', model_name)
    pilot_data['ModelType'] = kl.model_id()
    extract_data_from_pickles(cfg, tub_names, exclude=exclude)
    records = gather_records(cfg, tub_names, exclude=exclude,
                             data_base=pilot_data)
    if dry:
        print("Dry run only - stop here.\n")
        return

    gen_records = {}
    collate_records(records, gen_records, cfg)
    # random shuffle the records and reduce to size
    rec_list = list(gen_records.items())
    random.shuffle(rec_list)
    if train_frac:
        rec_list = rec_list[:int(train_frac * len(rec_list))]

    gen_records = dict(rec_list)
    shuffled_keys = list(gen_records.keys())
    count = 0
    # Ratio of samples to use as training data, the remaining are used for
    # evaluation
    train_count = int(cfg.TRAIN_TEST_SPLIT * len(shuffled_keys))
    for key in shuffled_keys:
        gen_records[key]['train'] = True
        count += 1
        if count >= train_count:
            break

    total_records = len(gen_records)
    print('Total records: {}, train: {}, validate: {}'
          .format(total_records, train_count, total_records - train_count))

    num_val = len(gen_records) - train_count
    train_gen = generator(kl, gen_records, cfg, True)
    val_gen = generator(kl, gen_records, cfg, False)

    steps_per_epoch = train_count // cfg.BATCH_SIZE
    val_steps = num_val // cfg.BATCH_SIZE
    assert val_steps > 0, "val steps > 0 required, please decrease batch " \
                          "size below {}".format(num_val)
    print('Steps_per_epoch', steps_per_epoch)
    cfg.model_type = model_type
    go_train(kl, cfg, train_gen, val_gen, model_name,
             steps_per_epoch, val_steps, pilot_data, verbose)


def handle_transfer(cfg, kl, pilot_data, transfer_model, tub_names):
    if not transfer_model:
        return tub_names
    assert (transfer_model[-3:] == '.h5')
    kl.load(transfer_model)
    print('Loading weights from model:', transfer_model, 'with ID:',
          kl.model_id())
    pilot_data['TransferModel'] = os.path.basename(transfer_model)
    # when transfering models, should we freeze all but the last N layers?
    if cfg.FREEZE_LAYERS:
        num_last_layers = getattr(cfg, 'NUM_LAST_LAYERS_TO_TRAIN', None)
        num_freeze = kl.freeze_first_layers(num_last_layers)
        pilot_data['Freeze'] = num_freeze
    # if transfer is given but no tubs, use tubs from transfer pilot
    if not tub_names:
        transfer_pilot_json = transfer_model.replace('.h5', '.json')
        assert os.path.exists(transfer_pilot_json), \
            "Can't train w/o tubs or transfer model data base."
        with open(transfer_pilot_json, 'r') as f:
            transfer_pilot = json.load(f)
            tub_names = transfer_pilot['Tubs']
    return tub_names


def go_train(kl, cfg, train_gen, val_gen, model_name,
             steps_per_epoch, val_steps, pilot_data, verbose):

    start = time.time()
    model_path = os.path.expanduser(model_name)
    # checkpoint to save model after each epoch and send best to the pi.
    save_best = MyCPCallback(filepath=model_path,
                             monitor='val_loss',
                             verbose=verbose,
                             save_best_only=True,
                             mode='min',
                             cfg=cfg,
                             pilot_data=pilot_data)

    # stop training if the validation error stops improving.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=cfg.MIN_DELTA,
                                               patience=cfg.EARLY_STOP_PATIENCE,
                                               verbose=verbose,
                                               mode='auto')

    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")

    epochs = cfg.MAX_EPOCHS
    workers_count = 1
    use_multiprocessing = False
    callbacks_list = [save_best]

    if cfg.USE_EARLY_STOP:
        callbacks_list.append(early_stop)

    if hasattr(cfg, 'USE_TENSORBOARD') and cfg.USE_TENSORBOARD:
        print("Using tensor board...")
        car_dir = os.path.dirname(os.path.realpath(__file__))
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(car_dir, "logs/fit/" + now)
        tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks_list.append(tb_cb)

    history = kl.model.fit(
                x=train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=cfg.VERBOSE_TRAIN,
                validation_data=val_gen,
                callbacks=callbacks_list,
                validation_steps=val_steps,
                workers=workers_count,
                use_multiprocessing=use_multiprocessing)

    duration_train = time.time() - start
    print("Training completed in %s."
          % str(datetime.timedelta(seconds=round(duration_train))))
    print("Best Eval Loss: %f" % save_best.best)
    print('-' * 100, '\n')

    if cfg.SHOW_PLOT:
        make_train_plot(history, model_path, save_best)


def convert_to_tflite(cfg, gen_records, model_path):
    # Save tflite, optionally in the int quant format for Coral TPU
    print("------------------- Saving TFLite Model -------------------")
    print('Reading pilot', model_path)
    tflite_fnm = model_path.replace(".h5", ".tflite")
    assert (".tflite" in tflite_fnm)
    # "coral" in cfg.model_type

    prepare_for_coral = False
    if prepare_for_coral:
        # compile a list of records to calibrate the quantization
        data_list = []
        max_items = 1000
        for key, _record in gen_records.items():
            data_list.append(_record)
            if len(data_list) == max_items:
                break

        stride = 1
        num_calibration_steps = len(data_list) // stride

        # a generator function to help train the quantizer with the
        # expected range of data from inputs
        def representative_dataset_gen():
            start = 0
            end = stride
            for _ in range(num_calibration_steps):
                batch_data = data_list[start:end]
                inputs = []

                for record in batch_data:
                    filename = record['image_path']
                    img_arr = load_scaled_image_arr(filename, cfg)
                    inputs.append(img_arr)

                start += stride
                end += stride

                # Get sample input data as a numpy array in a method of
                # your choosing.
                yield [np.array(inputs,
                                dtype=np.float32).reshape(stride,
                                                          cfg.TARGET_H,
                                                          cfg.TARGET_W,
                                                          cfg.TARGET_D)]
    else:
        representative_dataset_gen = None

    try:
        keras_model_to_tflite(model_path, tflite_fnm, representative_dataset_gen)
        if prepare_for_coral:
            print("Compile for Coral w: edgetpu_compiler", tflite_fnm)
            os.system("edgetpu_compiler " + tflite_fnm)
        print("Saved TFLite model:", tflite_fnm)
    except Exception as e:
        print('Conversion of', model_path, 'failed because:', e)
    finally:
        pass


def make_train_plot(history, model_path, save_best):
    if not do_plot:
        return

    plt.figure(1)
    time_stamp = datetime.datetime.now().isoformat(sep='_', timespec='minutes')
    # Only do accuracy if we have that data (categorical outputs)
    if 'angle_out_acc' in history.history:
        plt.subplot(121)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper right')

    # summarize history for acc
    if 'angle_acc' in history.history:
        plt.subplot(122)
        plt.plot(history.history['angle_acc'])
        plt.plot(history.history['val_angle_acc'])
        plt.title('model angle accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        # plt.legend(['train', 'validate'], loc='upper left')

    plt.savefig(model_path + '_loss_acc_{:}.{:}'.format(time_stamp,
                                                        figure_format))


def sequence_generator(kl, data, cfg):
    num_records = len(data)
    batch_size = cfg.BATCH_SIZE
    is_imu = False
    is_mem = type(kl) is WorldMemory

    while True:
        for offset in range(0, num_records, batch_size):
            batch_data = data[offset:offset+batch_size]
            if len(batch_data) != batch_size:
                break

            b_inputs_img = []
            b_inputs_imu = []
            b_inputs_drive = []
            y1 = []
            y2 = []

            for seq in batch_data:
                inputs_img = []
                inputs_imu = []
                inputs_drive = []
                num_images_target = len(seq)
                # for memory model sequence is one longer as we want to
                # predict the next latent vector, imu and drive vector

                for iRec, record in enumerate(seq):
                    # get image data if we don't already have it
                    if len(inputs_img) < num_images_target:
                        if record['img_data'] is None:
                            img_path = record['image_path']
                            img_arr = load_scaled_image_arr(img_path, cfg)
                            if img_arr is None:
                                break
                            if aug:
                                img_arr = augment_image(img_arr)
                            if cfg.CACHE_IMAGES:
                                record['img_data'] = img_arr
                        # for world memory model img_data contains latent vec
                        else:
                            img_arr = record['img_data']

                        imu_arr = None
                        if 'imu_array' in record:
                            is_imu = True
                            imu_arr = record['imu_array']
                        drive_arr = [record['angle'], record['throttle']]
                        # for world memory model input is latent vector and
                        # angle/throttle series up to last entry
                        if is_mem and len(inputs_img) < num_images_target - 1:
                            inputs_img.append(img_arr)
                            inputs_drive.append(drive_arr)
                        if not is_mem:
                            inputs_img.append(img_arr)
                            if is_imu:
                                inputs_imu.append(imu_arr)

                    # add the latest angle / throttle to observation data in
                    # normal sequence model, and the next data in memory model
                    if iRec == num_images_target - 1:
                        if is_mem:
                            y1.append(np.squeeze(img_arr))
                        else:
                            y1.append(drive_arr[0])
                            y2.append(drive_arr[1])

                b_inputs_img.append(inputs_img)
                b_inputs_imu.append(inputs_imu)
                b_inputs_drive.append(inputs_drive)

            if is_mem:
                X = [np.array(b_inputs_img), np.array(b_inputs_drive)]
                # add dummy to target for internal state model output
                y = [np.array(y1), np.zeros((batch_size, kl.units))]

            else:
                X = [np.array(b_inputs_img)]
                if is_imu:
                    X.append(np.array(b_inputs_imu))
                y = [np.array(y1), np.array(y2)]

            yield X, y


def sequence_train(cfg, tub_names, model_name, transfer_model, model_type,
                   aug, exclude=None, train_frac=None, dry=False):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    trains models which take sequence of images
    '''
    seq_step = getattr(cfg, 'SEQUENCE_TRAIN_STEP_SIZE', 1)
    print("Sequence of images training, using step size", seq_step)
    pilot_num = 0
    pilot_data = {}
    if model_name is None:
        model_name, pilot_num = auto_generate_model_name()
        model_name += ".h5"
    pilot_data['Num'] = pilot_num
    pilot_data['SeqStep'] = seq_step

    kl = dk.utils.get_model_by_type(model_type=model_type, cfg=cfg)
    tub_names = handle_transfer(cfg, kl, pilot_data, transfer_model, tub_names)
    kl.compile()
    print('Training new pilot:', model_name)
    pilot_data['ModelType'] = kl.model_id()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary(line_length=180))

    verbose = cfg.VERBOSE_TRAIN
    encoder = None
    if type(kl) is WorldMemory:
        encoder = kl.encoder
    records = gather_records(cfg, tub_names, exclude=exclude,
                             data_base=pilot_data, encoder=encoder)
    if dry:
        print("Dry run only - stop here.\n")
        return

    gen_records = {}
    collate_records(records, gen_records, cfg)
    # memory model requires sequence of 1 more as we forcast the next data
    extend_length = 1 if type(kl) is WorldMemory else 0
    sequences = collating_sequences(cfg, gen_records, extend_length)
    if train_frac:
        random.shuffle(sequences)
        sequences = sequences[:int(train_frac * len(sequences))]

    print("Collated", len(sequences), "sequences of length", cfg.SEQUENCE_LENGTH)
    # shuffle and split the data
    train_data, val_data \
        = train_test_split(sequences, test_size=(1 - cfg.TRAIN_TEST_SPLIT))

    train_gen = sequence_generator(kl, train_data, cfg)
    val_gen = sequence_generator(kl, val_data, cfg)
    total_train = len(train_data)
    total_val = len(val_data)

    steps_per_epoch = total_train // cfg.BATCH_SIZE
    val_steps = total_val // cfg.BATCH_SIZE
    print('Train: %d, validation: %d steps_per_epoch: %d'
          %(total_train, total_val, steps_per_epoch))
    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")

    cfg.model_type = model_type
    go_train(kl, cfg, train_gen, val_gen, model_name,
             steps_per_epoch, val_steps, pilot_data=pilot_data, verbose=verbose)


def collating_sequences(cfg, gen_records, extend_length=0):
    sequences = []
    target_len = cfg.SEQUENCE_LENGTH + extend_length
    step_size = getattr(cfg, 'SEQUENCE_TRAIN_STEP_SIZE', 1)
    assert type(step_size) is int, 'Sequence step size must be integer.'
    print('Collating {:} sequences with length {:} step size {:}'\
          .format(len(gen_records), target_len, step_size))
    for k, sample in tqdm(gen_records.items()):
        seq = []
        for i in range(target_len):
            key = make_next_key(sample, i * step_size)
            if key in gen_records:
                seq.append(gen_records[key])
            else:
                continue
        if len(seq) != target_len:
            continue
        sequences.append(seq)
    return sequences


def multi_train(cfg, tub, model, transfer, model_type, aug, exclude=None,
                train_frac=None, dry=False):
    '''
    choose the right regime for the given model type
    '''
    train_fn = train
    if model_type in ('rnn', '3d', 'look_ahead', 'square_plus_lstm',
                      'square_plus_imu_lstm', 'world_memory'):
        train_fn = sequence_train
    train_fn(cfg, tub, model, transfer, model_type, aug, exclude,
             train_frac, dry)


def prune(model, validation_generator, val_steps, cfg):
    pct_pruning = float(cfg.PRUNE_PERCENT_PER_ITERATION)
    total_channels = get_total_channels(model)
    n_channels_delete = int(math.floor(pct_pruning / 100 * total_channels))

    apoz_df = get_model_apoz(model, validation_generator)
    model = prune_model(model, apoz_df, n_channels_delete)
    name = '{}/model_pruned_{}_percent.h5'.format(cfg.MODELS_PATH, pct_pruning)
    model.save(name)

    return model, n_channels_delete


def extract_data_from_pickles(cfg, tubs, exclude=[]):
    """
    Extracts record_{id}.json and image from a pickle with the same id if
    exists in the tub.
    Then writes extracted json/jpg along side the source pickle that tub.
    This assumes the format {id}.pickle in the tub directory.
    :param cfg: config with data location configuration. Generally the global
    config object.
    :param tubs: The list of tubs involved in training.
    :param exclude: string patterns to exclude form tub path names
    :return: implicit None.
    """
    t_paths = gather_tub_paths(cfg, tubs, exclude)
    for tub_path in t_paths:
        file_paths = glob.glob(join(tub_path, '*.pickle'))
        print('Found {} pickles writing json records and images in tub {}'
              .format(len(file_paths), tub_path))
        for file_path in file_paths:
            # print('loading data from {}'.format(file_paths))
            with open(file_path, 'rb') as f:
                p = zlib.decompress(f.read())
            data = pickle.loads(p)

            base_path = dirname(file_path)
            filename = splitext(basename(file_path))[0]
            image_path = join(base_path, filename + '.jpg')
            img = Image.fromarray(np.uint8(data['val']['cam/image_array']))
            img.save(image_path)

            data['val']['cam/image_array'] = filename + '.jpg'

            with open(join(base_path, 'record_{}.json'.format(filename)), 'w') as f:
                json.dump(data['val'], f)


def prune_model(model, apoz_df, n_channels_delete):
    from kerassurgeon import Surgeon
    import pandas as pd

    # Identify 5% of channels with the highest APoZ in model
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'],
                                  dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name),
                        channels=channels)
    # Delete channels
    return surgeon.operate()


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


def get_model_apoz(model, generator):
    from kerassurgeon.identify import get_apoz
    import pandas as pd

    # Get APoZ
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df


def remove_comments(dir_list):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]


def preprocessFileList(filelist):
    dirs = []
    if filelist is not None:
        for afile in filelist:
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                dirs.extend(tmp_dirs)

    remove_comments(dirs)
    return dirs


def auto_generate_model_name():
    """ Assumes models are in directory models/ and their names are
    pilot_N_YY_MM_DD, where N is a continuous counter. """
    car_dir = os.getcwd()
    model_path = os.path.expanduser(os.path.join(car_dir, 'models'))
    assert os.path.exists(model_path), model_path + " not found"
    print('Found model path', model_path)
    files = os.listdir(model_path)
    model_files = [f for f in files if f[-3:] == '.h5']
    pilot_nums = [0]
    for model_file in model_files:
        pilot_path = os.path.basename(model_file)
        # splitting off extension '.h5'
        pilot_split = pilot_path.split('.')
        if len(pilot_split) == 2:
            pilot_name = pilot_split[0]
            pilot_name_split = pilot_name.split('_')
            # this is true if name is 'pilot_XYZ_YY-MM-DD' or any other
            # string after the number XYZ
            if len(pilot_name_split) == 3 and pilot_name_split[0] == 'pilot' \
                    and pilot_name_split[1].isnumeric():
                pilot_num = int(pilot_name_split[1])
                pilot_nums.append(pilot_num)

    new_pilot_num = max(pilot_nums) + 1
    new_pilot = 'models/pilot_' + str(new_pilot_num) + '_' + \
                datetime.datetime.now().strftime('%y-%m-%d')
    return new_pilot, new_pilot_num


def train_for_remote(cfg, model_type):
    """
    Method for continuous training of live tub on the car. Assumes, that the
    last tub is currently being live. The method will continuously cycle
    through rsyncing the last tub, training the pilot and rsyncing the pilot
    back to the car when training finished. The pilot will be named
    pilot_continuous.tflite.

    :param cfg: car configuration
    :return:    None
    """

    pi_hostname = cfg.PI_HOSTNAME
    pi_user = cfg.PI_USERNAME
    remote_car_dir = cfg.PI_DONKEY_ROOT
    remote_tub_dir = os.path.join(remote_car_dir, 'data')
    local_file = 'donkey_tubs.list'
    # get the remote dir listing as a local file
    command = 'ssh ' + pi_user + '@' + pi_hostname + ' ls ' + remote_tub_dir \
              + ' > ' + local_file
    print('Executing', command)
    out = os.system(command)
    if out != 0:
        print('Cannot ssh into ' + pi_hostname + ' with user ' + pi_user +
              ' or directory ' + remote_tub_dir + ' does not exist.')
        return
    # now read tub list file
    with open(local_file) as f:
        tub_list = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    tub_list = [x.strip() for x in tub_list]
    # find latest tub on the remote:
    th = TubHandler('data')
    last_tub_num, last_tub = th.get_last_tub(tub_list)
    # local tub path
    tub_path = os.path.join('data', last_tub)
    # rsync get command
    rsync_get = 'rsync -a ' + pi_user + '@' + pi_hostname + ':' + \
                os.path.join(remote_tub_dir, last_tub) + ' data'
    # hardcode pilot name for now
    pilot = 'models/pilot_continuous.h5'
    pilot_tflite = pilot.replace('.h5', '.tflite')
    # rsync put command
    rsync_put = 'rsync ' + pilot_tflite + ' ' \
                + pi_user + '@' + pi_hostname + ':' \
                + os.path.join(remote_car_dir, pilot_tflite)
    # overwrite parameters for fast on-the-fly training
    cfg.MAX_EPOCHS = 20
    cfg.EARLY_STOP_PATIENCE = 5

    while True:
        try:
            # get tub from car
            rsync_success(rsync_get)
            # use previous result as transfer
            transfer_model = pilot if os.path.exists(pilot) else None
            # tub check first
            tub = Tub(tub_path)
            tub.check(fix=True)
            # for < 10k records wait a bit and try again
            num_rec = tub.get_num_records()
            if num_rec < 1e4:
                print('Retrieved only {:} records, waiting for 30s and try '
                      'again.'.format(num_rec))
                time.sleep(30)
                continue
            # now train pilot
            train(cfg, tub_path, pilot, transfer_model, model_type, aug=False)
            # convert to tflite
            convert_to_tflite(cfg, {}, pilot)
            # put pilot back
            rsync_success(rsync_put)

        except KeyboardInterrupt:
            print('User interrupt')
            break

        except OSError as e:
            print(e)
            break


def rsync_success(command):
    print('Executing', command, end='')
    tic = time.time()
    out = os.system(command)
    if out != 0:
        raise OSError('Failed to rsync')
    toc = time.time()
    print(' ... in {:.1f}s'.format(toc-tic))


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config()
    convert = args['convert']
    remote = args['remote']
    tub = args['--tub']
    exclude = args['--exclude']
    model = args['--model']
    transfer = args['--transfer']
    model_type = args['--type']
    if args['--figure_format']:
        figure_format = args['--figure_format']
    aug = args['--aug']
    nn_size = args['--nn_size']
    train_frac = args['--frac']
    dry = args['--dry']

    if nn_size is not None:
        cfg.NN_SIZE = nn_size

    if convert:
        model_dir = os.path.join(os.getcwd(), 'models')
        files = os.listdir(model_dir)
        pilots_h5 = [os.path.join(model_dir, f) for f in files if f[-3:] ==
                     '.h5']
        for pilot_h5 in pilots_h5:
            if os.path.exists(pilot_h5.replace('.h5', '.tflite')):
                continue
            convert_to_tflite(cfg, {}, pilot_h5)

    elif remote:
        train_for_remote(cfg, model_type=model_type)

    else:
        dirs = preprocessFileList(args['--file'])
        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend(tub_paths)
        # switched off b/c of incompatibility with Lstm
        # disable_eager_execution()
        train_frac = float(train_frac) if train_frac else None
        multi_train(cfg, dirs, model, transfer, model_type, aug, exclude,
                    train_frac, dry)

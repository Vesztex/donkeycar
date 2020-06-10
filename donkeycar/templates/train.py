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
    train.py [--tub=<tub1,tub2,..tubn>]
    [--exclude=<[pattern1, pattern2]>]
    [--file=<file> ...]
    [--model=<model>]
    [--transfer=<model>]
    [--type=(linear|latent|categorical|rnn|imu|behavior|3d|look_ahead|tensorrt_linear|tflite_linear|coral_tflite_linear)]
    [--figure_format=<figure_format>]
    [--nn_size=<nn_size>]
    [--continuous] [--aug] [--dry]

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
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from docopt import docopt

import donkeycar as dk
from donkeycar.parts.keras import KerasIMU, KerasCategorical, KerasBehavioral, \
    KerasLatent, KerasLocalizer, KerasSquarePlusImu
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


def collate_records(records, gen_records, opts):
    '''
    open all the .json records from records list passed in, read their contents,
    add them to a list of gen_records, passed in. use the opts dict to
    specify config choices
    '''
    print('Collating %d records ...' % (len(records)))
    throttle_key = 'user/throttle'
    if hasattr(opts['cfg'], 'USE_SPEED_FOR_MODEL') \
            and opts['cfg'].USE_SPEED_FOR_MODEL:
        throttle_key = 'car/speed'

    print('Using', throttle_key, 'for training')
    new_records = {}

    for record_path in tqdm(records):

        basepath = os.path.dirname(record_path)
        index = get_record_index(record_path)
        sample = {'tub_path': basepath, "index": index}

        key = make_key(sample)

        if key in gen_records:
            continue

        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except:
            continue

        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)

        sample['record_path'] = record_path
        sample["image_path"] = image_path
        sample["json_data"] = json_data

        angle = float(json_data['user/angle'])
        throttle = float(json_data[throttle_key])

        if opts['categorical']:
            r = opts['cfg'].MODEL_CATEGORICAL_MAX_THROTTLE_RANGE
            angle = dk.utils.linear_bin(angle)
            throttle = dk.utils.linear_bin(throttle, N=20, offset=0, R=r)

        sample['angle'] = angle
        sample['throttle'] = throttle

        try:
            accl_x = float(json_data['imu/acl_x'])
            accl_y = float(json_data['imu/acl_y'])
            accl_z = float(json_data['imu/acl_z'])

            gyro_x = float(json_data['imu/gyr_x'])
            gyro_y = float(json_data['imu/gyr_y'])
            gyro_z = float(json_data['imu/gyr_z'])

            sample['imu_array'] = np.array([accl_x, accl_y, accl_z,
                                            gyro_x, gyro_y, gyro_z])
        except KeyError:
            pass

        try:
            accel = json_data['car/accel']
            gyro = json_data['car/gyro']
            sample['imu_array'] = np.array(accel + gyro)
        except KeyError:
            pass

        try:
            behavior_arr = np.array(json_data['behavior/one_hot_state_array'])
            sample["behavior_arr"] = behavior_arr
        except KeyError:
            pass

        try:
            location_arr = np.array(json_data['location/one_hot_state_array'])
            sample["location"] = location_arr
        except KeyError:
            pass

        sample['img_data'] = None

        # Initialise 'train' to False
        sample['train'] = False

        # We need to maintain the correct train - validate ratio across the
        # dataset, even if continous training so don't add this sample to the
        # main records list (gen_records) yet.
        new_records[key] = sample

    # new_records now contains all our NEW samples - set a random selection
    # to be the training samples based on the ratio in CFG file
    shufKeys = list(new_records.keys())
    random.shuffle(shufKeys)
    trainCount = 0
    # Ratio of samples to use as training data, the remaining are used for
    # evaluation
    targetTrainCount = int(opts['cfg'].TRAIN_TEST_SPLIT * len(shufKeys))
    for key in shufKeys:
        new_records[key]['train'] = True
        trainCount += 1
        if trainCount >= targetTrainCount:
            break
    # Finally add all the new records to the existing list
    gen_records.update(new_records)


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

    def __init__(self, send_model_cb=None, cfg=None, *args, **kwargs):
        super(MyCPCallback, self).__init__(*args, **kwargs)
        self.reset_best_end_of_epoch = False
        self.send_model_cb = send_model_cb
        self.last_modified_time = None
        self.cfg = cfg

    def reset_best(self):
        self.reset_best_end_of_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        super(MyCPCallback, self).on_epoch_end(epoch, logs)

        if self.send_model_cb:
            '''
            check whether the file changed and send to the pi
            '''
            filepath = self.filepath.format(epoch=epoch, **logs)
            if os.path.exists(filepath):
                last_modified_time = os.path.getmtime(filepath)
                if self.last_modified_time is None or self.last_modified_time < last_modified_time:
                    self.last_modified_time = last_modified_time
                    self.send_model_cb(self.cfg, self.model, filepath)

        '''
        when reset best is set, we want to make sure to run an entire epoch
        before setting our new best on the new total records
        '''
        if self.reset_best_end_of_epoch:
            self.reset_best_end_of_epoch = False
            self.best = np.Inf


def on_best_model(cfg, model, model_filename):

    model.save(model_filename, include_optimizer=False)

    if not cfg.SEND_BEST_MODEL_TO_PI:
        return

    on_windows = os.name == 'nt'

    # If we wish, send the best model to the pi.
    # On mac or linux we have scp:
    if not on_windows:
        print('sending model to the pi')

        command = 'scp %s %s@%s:~/%s/models/;' % (model_filename, cfg.PI_USERNAME, cfg.PI_HOSTNAME, cfg.PI_DONKEY_ROOT)

        print("sending", command)
        res = os.system(command)
        print(res)

    else:  # yes, we are on windows machine

        #On windoz no scp. In order to use this you must first setup
        #an ftp daemon on the pi. ie. sudo apt-get install vsftpd
        #and then make sure you enable write permissions in the conf
        try:
            import paramiko
        except:
            raise Exception("first install paramiko: pip install paramiko")

        host = cfg.PI_HOSTNAME
        username = cfg.PI_USERNAME
        password = cfg.PI_PASSWD
        server = host
        files = []

        localpath = model_filename
        remotepath = '/home/%s/%s/%s' %(username, cfg.PI_DONKEY_ROOT, model_filename.replace('\\', '/'))
        files.append((localpath, remotepath))

        print("sending", files)

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
            ssh.connect(server, username=username, password=password)
            sftp = ssh.open_sftp()

            for localpath, remotepath in files:
                sftp.put(localpath, remotepath)

            sftp.close()
            ssh.close()
            print("send succeded")
        except:
            print("send failed")


def train(cfg, tub_names, model_name, transfer_model,
          model_type, continuous, aug, exclude=None, dry=False):
    """
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    """
    verbose = cfg.VERBOSE_TRAIN
    pilot_num = 0
    pilot_data = {}
    if model_name is None:
        model_name, pilot_num = auto_generate_model_name()
        model_name += ".tflite" if "tflite" in model_type else ".uff" if \
            "tensorrt" in model_type else ".h5"
    pilot_data['Num'] = pilot_num

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE

    if "tflite" in model_type:
        # even though we are passed the .tflite output file, we train with an
        # intermediate .h5 output and then convert to final .tflite at the end.
        assert(".tflite" in model_name)
        # we only support linear or square+ model type right now for tflite
        assert("linear" in model_type or "square_plus")
        model_name = model_name.replace(".tflite", ".h5")
    elif "tensorrt" in model_type:
        # even though we are passed the .uff output file, we train with an
        # intermediate .h5 output and then convert to final .uff at the end.
        assert(".uff" in model_name)
        # we only support the linear model type right now for tensorrt
        assert("linear" in model_type)
        model_name = model_name.replace(".uff", ".h5")

    if model_name and not '.h5' == model_name[-3:]:
        raise Exception("Model filename should end with .h5")

    if continuous:
        print("continuous training")

    if aug:
        print("Using data augmentation")

    gen_records = {}
    opts = {'cfg': cfg}

    if "linear" in model_type:
        train_type = "linear"
    elif "square_plus_imu" in model_type:
        train_type = "square_plus_imu"
    elif "square_plus" in model_type:
        train_type = "square_plus"
    else:
        train_type = model_type

    kl = get_model_by_type(train_type, cfg=cfg)
    opts['categorical'] = type(kl) in [KerasCategorical, KerasBehavioral]
    print('Training with model type', kl.model_id())

    if transfer_model:
        assert(transfer_model[-3:] == '.h5')
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
            if os.path.exists(transfer_pilot_json):
                with open(transfer_pilot_json, 'r') as f:
                    transfer_pilot = json.load(f)
                    tub_names = transfer_pilot['Tubs']

    if cfg.OPTIMIZER:
        kl.set_optimizer(cfg.OPTIMIZER, cfg.LEARNING_RATE, cfg.LEARNING_RATE_DECAY)

    kl.compile()
    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    print('Training new pilot:', model_name)
    pilot_data['ModelType'] = kl.model_id()
    opts['keras_pilot'] = kl
    opts['continuous'] = continuous
    opts['model_type'] = model_type

    extract_data_from_pickles(cfg, tub_names, exclude=exclude)
    records = gather_records(cfg, tub_names, exclude=exclude,
                             verbose=True, data_base=pilot_data)

    if dry:
        print("Dry run only - stop here.\n")
        return

    collate_records(records, gen_records, opts)
    def generator(save_best, opts, data, batch_size, is_train_set=True,
                  min_records_to_train=1000):

        num_records = len(data)
        while True:
            if is_train_set and opts['continuous']:
                # When continuous training, we look for new records after each
                # epoch. This will add new records to the train and validation
                # set.
                records = gather_records(cfg, tub_names)
                if len(records) > num_records:
                    collate_records(records, gen_records, opts)
                    new_num_rec = len(data)
                    if new_num_rec > num_records:
                        new_rec = new_num_rec - num_records
                        print('picked up', new_rec, 'new records!')
                        num_records = new_num_rec
                        save_best.reset_best()
                if num_records < min_records_to_train:
                    print("not enough records to train. need %d, have %d. "
                          "waiting..." % (min_records_to_train, num_records))
                    time.sleep(10)
                    continue

            batch_data = []
            keys = list(data.keys())
            random.shuffle(keys)

            kl = opts['keras_pilot']
            if type(kl.model.output) is list:
                model_out_shape = (2, 1)
            else:
                model_out_shape = kl.model.output.shape

            has_imu = type(kl) is KerasIMU or type(kl) is KerasSquarePlusImu
            has_bvh = type(kl) is KerasBehavioral
            img_out = type(kl) is KerasLatent
            loc_out = type(kl) is KerasLocalizer
            imu_dim = cfg.IMU_DIM if hasattr(cfg, 'IMU_DIM') else 6

            if img_out:
                import cv2

            for key in keys:
                if not key in data:
                    continue
                _record = data[key]
                if _record['train'] != is_train_set:
                    continue
                if continuous:
                    # in continuous mode we need to handle files getting deleted
                    filename = _record['image_path']
                    if not os.path.exists(filename):
                        data.pop(key, None)
                        continue

                batch_data.append(_record)
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

                    img_arr = np.array(inputs_img)\
                        .reshape(batch_size, cfg.TARGET_H, cfg.TARGET_W,
                                 cfg.TARGET_D)

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
                        y = [np.array([out]).reshape(batch_size, 2) ]
                    else:
                        y = [np.array(angles), np.array(throttles)]

                    yield X, y

                    batch_data = []

    model_path = os.path.expanduser(model_name)

    # checkpoint to save model after each epoch and send best to the pi.
    save_best = MyCPCallback(send_model_cb=on_best_model,
                             filepath=model_path,
                             monitor='val_loss',
                             verbose=verbose,
                             save_best_only=True,
                             mode='min',
                             cfg=cfg)

    train_gen = generator(save_best, opts, gen_records, cfg.BATCH_SIZE, True)
    val_gen = generator(save_best, opts, gen_records, cfg.BATCH_SIZE, False)
    total_records = len(gen_records)

    num_train = 0
    num_val = 0
    for key, _record in gen_records.items():
        if _record['train']:
            num_train += 1
        else:
            num_val += 1

    print('Total records: {}, train: {}, validate: {}'
          .format(total_records, num_train, num_val))

    steps_per_epoch = 100 if continuous else num_train // cfg.BATCH_SIZE
    val_steps = num_val // cfg.BATCH_SIZE
    print('Steps_per_epoch', steps_per_epoch)
    cfg.model_type = model_type
    go_train(kl, cfg, train_gen, val_gen, gen_records, model_name,
             steps_per_epoch, val_steps, continuous, verbose, save_best)

    pilot_data['Accuracy'] = save_best.best
    pilot_data['Date'] = datetime.datetime.now().date().isoformat()
    # save model data
    with open(model_name.replace('.h5', '.json'), 'w') as f:
        json.dump(pilot_data, f)


def go_train(kl, cfg, train_gen, val_gen, gen_records, model_name,
             steps_per_epoch, val_steps, continuous, verbose, save_best=None):

    start = time.time()

    model_path = os.path.expanduser(model_name)

    # checkpoint to save model after each epoch and send best to the pi.
    if save_best is None:
        save_best = MyCPCallback(send_model_cb=on_best_model,
                                 filepath=model_path, monitor='val_loss',
                                 verbose=verbose, save_best_only=True,
                                 mode='min', cfg=cfg)

    # stop training if the validation error stops improving.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=cfg.MIN_DELTA,
                                                patience=cfg.EARLY_STOP_PATIENCE,
                                                verbose=verbose,
                                                mode='auto')

    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")

    epochs = 100000 if continuous else cfg.MAX_EPOCHS
    workers_count = 1
    use_multiprocessing = False
    callbacks_list = [save_best]

    if cfg.USE_EARLY_STOP and not continuous:
        callbacks_list.append(early_stop)

    if hasattr(cfg, 'USE_TENSORBOARD') and cfg.USE_TENSORBOARD:
        print("Using tensor board...")
        car_dir = os.path.dirname(os.path.realpath(__file__))
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(car_dir, "logs/fit/" + now)
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=1)
        callbacks_list.append(tb_cb)

    history = kl.model.fit_generator(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=cfg.VERBOSE_TRAIN,
                    validation_data=val_gen,
                    callbacks=callbacks_list,
                    validation_steps=val_steps,
                    workers=workers_count,
                    use_multiprocessing=use_multiprocessing)

    full_model_val_loss = min(history.history['val_loss'])
    max_val_loss = full_model_val_loss + cfg.PRUNE_VAL_LOSS_DEGRADATION_LIMIT

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))) )
    print("Best Eval Loss: %f" % save_best.best)

    if cfg.SHOW_PLOT:
        try:
            if do_plot:
                plt.figure(1)

                # Only do accuracy if we have that data (e.g. categorical outputs)
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
                if 'angle_out_acc' in history.history:
                    plt.subplot(122)
                    plt.plot(history.history['angle_out_acc'])
                    plt.plot(history.history['val_angle_out_acc'])
                    plt.title('model angle accuracy')
                    plt.ylabel('acc')
                    plt.xlabel('epoch')
                    # plt.legend(['train', 'validate'], loc='upper left')

                plt.savefig(model_path + '_loss_acc_%f.%s' % (save_best.best,
                                                              figure_format))
                plt.show(block=False)
            else:
                print("not saving loss graph because matplotlib not set up.")
        except Exception as ex:
            print("problems with loss graph: {}".format(ex))

    # Save tflite, optionally in the int quant format for Coral TPU
    if "tflite" in cfg.model_type:
        print("--------- Saving TFLite Model ---------")
        tflite_fnm = model_path.replace(".h5", ".tflite")
        assert(".tflite" in tflite_fnm)

        prepare_for_coral = "coral" in cfg.model_type

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

        from donkeycar.parts.tflite import keras_model_to_tflite
        keras_model_to_tflite(model_path, tflite_fnm, representative_dataset_gen)
        print("Saved TFLite model:", tflite_fnm)
        if prepare_for_coral:
            print("compile for Coral w: edgetpu_compiler", tflite_fnm)
            os.system("edgetpu_compiler " + tflite_fnm)

    # Save tensorrt
    if "tensorrt" in cfg.model_type:
        print("\n\n--------- Saving TensorRT Model ---------")
        # TODO RAHUL
        # flatten model_path
        # convert to uff
        # print("Saved TensorRT model:", uff_filename)


def sequence_train(cfg, tub_names, model_name, transfer_model, model_type,
                   continuous, aug, exclude=None, dry=False):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    trains models which take sequence of images
    '''
    assert(not continuous)
    print("Sequence of images training")

    kl = dk.utils.get_model_by_type(model_type=model_type, cfg=cfg)
    tubs = gather_tubs(cfg, tub_names, exclude)
    verbose = cfg.VERBOSE_TRAIN
    records = []

    for tub in tubs:
        record_paths = glob.glob(os.path.join(tub.path, 'record_*.json'))
        print("Tub:", tub.path, "has", len(record_paths), 'records')
        record_paths.sort(key=get_record_index)
        records += record_paths

    gen_records = {}
    throttle_key = 'user/throttle'
    if hasattr(cfg, 'USE_SPEED_FOR_MODEL') and cfg.USE_SPEED_FOR_MODEL:
        throttle_key = 'car/speed'
    if dry:
        print("Dry run only - stop here.\n")
        return

    print('Collating records, using', throttle_key, 'for training:')
    for record_path in tqdm(records):

        with open(record_path, 'r') as fp:
            json_data = json.load(fp)

        basepath = os.path.dirname(record_path)
        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)
        sample = {'record_path': record_path, 'image_path': image_path,
                  'json_data': json_data, "tub_path": basepath,
                  "index": get_image_index(image_filename)}

        angle = float(json_data['user/angle'])
        throttle = float(json_data[throttle_key])

        sample['target_output'] = np.array([angle, throttle])
        sample['angle'] = angle
        sample['throttle'] = throttle
        sample['img_data'] = None

        key = make_key(sample)
        gen_records[key] = sample

    print('Collating sequences')
    sequences = []
    target_len = cfg.SEQUENCE_LENGTH
    look_ahead = False

    if model_type == "look_ahead":
        target_len = cfg.SEQUENCE_LENGTH * 2
        look_ahead = True

    for k, sample in gen_records.items():
        seq = []
        for i in range(target_len):
            key = make_next_key(sample, i)
            if key in gen_records:
                seq.append(gen_records[key])
            else:
                continue

        if len(seq) != target_len:
            continue

        sequences.append(seq)

    print("Collated", len(sequences), "sequences of length", target_len)
    # shuffle and split the data
    train_data, val_data \
        = train_test_split(sequences, test_size=(1 - cfg.TRAIN_TEST_SPLIT))

    def generator(data, opt, batch_size=cfg.BATCH_SIZE):
        num_records = len(data)
        while True:
            # shuffle again for good measure
            random.shuffle(data)
            for offset in range(0, num_records, batch_size):
                batch_data = data[offset:offset+batch_size]
                if len(batch_data) != batch_size:
                    break

                b_inputs_img = []
                b_vec_in = []
                b_labels = []

                for seq in batch_data:
                    inputs_img = []
                    vec_in = []
                    labels = []
                    vec_out = []
                    num_images_target = len(seq)
                    i_target_out = -1
                    if opt['look_ahead']:
                        num_images_target = cfg.SEQUENCE_LENGTH
                        i_target_out = cfg.SEQUENCE_LENGTH - 1

                    for iRec, record in enumerate(seq):
                        # get image data if we don't already have it
                        if len(inputs_img) < num_images_target:
                            if record['img_data'] is None:
                                img_arr = load_scaled_image_arr(record['image_path'], cfg)
                                if img_arr is None:
                                    break
                                if aug:
                                    img_arr = augment_image(img_arr)
                                if cfg.CACHE_IMAGES:
                                    record['img_data'] = img_arr
                            else:
                                img_arr = record['img_data']

                            inputs_img.append(img_arr)

                        if iRec >= i_target_out:
                            vec_out.append(record['angle'])
                            vec_out.append(record['throttle'])
                        else:
                            vec_in.append(0.0)  # record['angle'])
                            vec_in.append(0.0)  # record['throttle'])

                    label_vec = seq[i_target_out]['target_output']

                    if look_ahead:
                        label_vec = np.array(vec_out)

                    labels.append(label_vec)
                    b_inputs_img.append(inputs_img)
                    b_vec_in.append(vec_in)
                    b_labels.append(labels)

                if look_ahead:
                    X = [np.array(b_inputs_img).reshape(batch_size,
                                                        cfg.TARGET_H,
                                                        cfg.TARGET_W,
                                                        cfg.SEQUENCE_LENGTH),
                         np.array(b_vec_in)]
                    y = np.array(b_labels).reshape(batch_size,
                                                   (cfg.SEQUENCE_LENGTH + 1) * 2)
                else:
                    X = [np.array(b_inputs_img).reshape(batch_size,
                                                        cfg.SEQUENCE_LENGTH,
                                                        cfg.TARGET_H,
                                                        cfg.TARGET_W,
                                                        cfg.TARGET_D)]
                    y = np.array(b_labels).reshape(batch_size, 2)

                yield X, y

    opt = {'look_ahead': look_ahead, 'cfg': cfg}
    train_gen = generator(train_data, opt)
    val_gen = generator(val_data, opt)
    total_train = len(train_data)
    total_val = len(val_data)

    print('train: %d, validation: %d' %(total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    val_steps = total_val // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)
    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")

    cfg.model_type = model_type
    go_train(kl, cfg, train_gen, val_gen, gen_records, model_name,
             steps_per_epoch, val_steps, continuous, verbose)


def multi_train(cfg, tub, model, transfer, model_type, continuous, aug,
                exclude=None, dry=False):
    '''
    choose the right regime for the given model type
    '''
    train_fn = train
    if model_type in ('rnn', '3d', "look_ahead"):
        train_fn = sequence_train

    train_fn(cfg, tub, model, transfer, model_type, continuous, aug, exclude,
             dry)


def prune(model, validation_generator, val_steps, cfg):
    percent_pruning = float(cfg.PRUNE_PERCENT_PER_ITERATION)
    total_channels = get_total_channels(model)
    n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))

    apoz_df = get_model_apoz(model, validation_generator)
    model = prune_model(model, apoz_df, n_channels_delete)
    name = '{}/model_pruned_{}_percent.h5'.format(cfg.MODELS_PATH, percent_pruning)
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
    car_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.expanduser(os.path.join(car_dir, 'models'))
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


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config()
    tub = args['--tub']
    exclude = args['--exclude']
    model = args['--model']
    transfer = args['--transfer']
    model_type = args['--type']
    if args['--figure_format']:
        figure_format = args['--figure_format']
    continuous = args['--continuous']
    aug = args['--aug']
    nn_size = args['--nn_size']
    dry = args['--dry']
    if nn_size is not None:
        cfg.NN_SIZE = nn_size

    dirs = preprocessFileList(args['--file'])
    if tub is not None:
        tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
        dirs.extend(tub_paths)

    multi_train(cfg, dirs, model, transfer, model_type, continuous, aug,
                exclude, dry)

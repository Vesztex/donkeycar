'''

pilots.py

Methods to create, use, save and load pilots. Pilots
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more
models to help direct the vehicles motion.

'''
import copy
import time

import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, Conv2D, MaxPooling2D,\
    AveragePooling2D, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Conv2DTranspose

import donkeycar as dk
from donkeycar.parts.tflite import TFLitePilot

if tf.__version__ == '1.13.1':
    from tensorflow import ConfigProto, Session

    # Override keras session to work around a bug in TF 1.13.1
    # Remove after we upgrade to TF 1.14 / TF 2.x.
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    keras.backend.set_session(session)


class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to
    guide a car.
    '''
    def __init__(self):
        self.model = None
        self.optimizer = "adam"

    def load(self, model_path):
        prev = self.model_id()
        self.model = keras.models.load_model(model_path)
        print("Load model: overwriting " + prev + " with " + self.model_id())

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        if optimizer_type == "adam":
            self.model.optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            self.model.optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            self.model.optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)

    def get_input_shape(self):
        assert self.model is not None, "Need to load model first"
        return self.model.inputs[0].shape

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=15, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor='val_loss',
                                                    verbose=verbose,
                                                    save_best_only=True,
                                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=min_delta,
                                                   patience=patience,
                                                   verbose=verbose,
                                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit(
                        x=train_gen,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        validation_steps=steps*(1.0 - train_split))
        return hist

    def get_num_last_layers_to_train(self):
        """ Find the canonically named Flatten layer and return number of
        layers after that"""
        i = 0
        while self.model.layers[i].name != 'flattened':
            i += 1
        return len(self.model.layers) - i - 1

    def freeze_first_layers(self, num_last_layers_to_train=None):
        if num_last_layers_to_train is None:
            num_last_layers_to_train = self.get_num_last_layers_to_train()
        num_to_freeze = len(self.model.layers) - num_last_layers_to_train
        frozen_layers = []
        for i in range(num_to_freeze):
            self.model.layers[i].trainable = False
            frozen_layers.append(self.model.layers[i].name)
        print('Freezing layers {}'.format(frozen_layers))
        return num_to_freeze

    def model_id(self):
        if self.model is None:
            return 'Model_not_set'
        return self.model.name

    def update(self, keras_pilot):
        if keras_pilot is None:
            return
        assert isinstance(keras_pilot, KerasPilot), \
            'Can only update KerasPilot from KerasPilot but not from ' \
            + type(keras_pilot).__name__
        self.model = keras_pilot.model


class KerasCategorical(KerasPilot):
    '''
    The KerasCategorical pilot breaks the steering and throttle decisions into discreet
    angles and then uses categorical cross entropy to train the network to activate a single
    neuron for each steering and throttle choice. This can be interesting because we
    get the confidence value as a distribution over all choices.
    This uses the dk.utils.linear_bin and dk.utils.linear_unbin to transform continuous
real numbers into a range of discreet values for training and runtime.
    The input and output are therefore bounded and must be chosen wisely to match the data.
    The default ranges work for the default setup. But cars which go faster may want to
    enable a higher throttle range. And cars with larger steering throw may want more bins.
    '''
    def __init__(self, input_shape=(120, 160, 3), throttle_range=0.5,
                 roi_crop=(0, 0), *args, **kwargs):
        super().__init__()
        self.model = default_categorical(input_shape, roi_crop)
        self.compile()
        self.throttle_range = throttle_range

    def compile(self):
        self.model.compile(optimizer=self.optimizer, metrics=['acc'],
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'categorical_crossentropy'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 1.0})

    def run(self, img_arr):
        if img_arr is None:
            print('no image')
            return 0.0, 0.0

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        N = len(throttle[0])
        throttle = dk.utils.linear_unbin(throttle, N=N, offset=0.0, R=self.throttle_range)
        angle_unbinned = dk.utils.linear_unbin(angle_binned)
        return angle_unbinned, throttle


class KerasLinear(KerasPilot):
    '''
    The KerasLinear pilot uses one neuron to output a continous value via the
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3),
                 roi_crop=(0, 0)):
        super().__init__()
        self.model = default_n_linear(num_outputs, input_shape, roi_crop)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasSquarePlus(KerasLinear):
    """
    improved cnn over standard keraslinear. the cnn translates into square
    matrices from layer one on and uses batch normalisation, average pooling
    & l2  regularisor instead of dropout in the perceptron layers. because of
    the square form, the first layers stride need to be 3x4 (as this is the
    input picture h x w ratio). with this, the reduction in the first layer is
    larger, hence the whole model only has 5 cnn layers.
    """
    def __init__(self, input_shape=(120, 160, 3), roi_crop=(0, 0),
                 *args, **kwargs):
        self.size = kwargs.get('size', 'S').upper()
        self.model = self.make_model(input_shape, roi_crop)
        self.compile()
        print('Created', self.__class__.__name__, 'NN size:', self.size)

    def compile(self):
        self.model.compile(optimizer='adam', loss='mse')

    def make_model(self, input_shape, roi_crop):
        return linear_square_plus(input_shape, roi_crop, size=self.size)


class KerasSquarePlusLstm(KerasSquarePlus):
    """
    LSTM version of square plus model
    """
    def __init__(self, input_shape=(120, 160, 3), roi_crop=(0, 0),
                 *args, **kwargs):
        self.seq_length = kwargs.get('seq_length', 3)
        super().__init__(input_shape, roi_crop=roi_crop, *args, **kwargs)
        self.img_seq = []

    def make_model(self, input_shape, roi_crop):
        return linear_square_plus(input_shape, roi_crop,
                                  size=self.size,
                                  seq_len=self.seq_length)

    def run(self, img_arr):
        # if buffer empty fill to length
        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)
        # pop oldest img from front and append current img at end
        self.img_seq.pop(0)
        self.img_seq.append(img_arr)
        # reshape and run model
        new_shape = (1, self.seq_length, ) + img_arr.shape
        img_arr = np.array(self.img_seq).reshape(new_shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasSquarePlusImu(KerasSquarePlus):
    """
    The model is a variation of the SquarePlus model that also uses imu data
    """
    def __init__(self, input_shape=(120, 160, 3), roi_crop=(0, 0), *args, **kwargs):
        self.imu_dim = kwargs.get('imu_dim', 6)
        self.size = kwargs.get('size', 'S').upper()
        self.model = linear_square_plus_imu(input_shape, roi_crop,
                                            imu_dim=self.imu_dim,
                                            size=self.size)
        self.compile()
        print('Created', self.__class__.__name__, 'imu_dim:', self.imu_dim,
              'NN size:', self.size)

    def run(self, img_arr, imu=None):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        if imu is None:
            imu_arr = np.zeros((1, 6))
        else:
            imu_arr = np.array(imu)
        outputs = self.model.predict(x=[img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasIMU(KerasPilot):
    '''
    A Keras part that take an image and IMU vector as input,
    outputs steering and throttle

    Note: When training, you will need to vectorize the input from the IMU.
    Depending on the names you use for imu records, something like this will work:

    X_keys = ['cam/image_array','imu_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(rec):
        rec['imu_array'] = np.array([ rec['imu/acl_x'], rec['imu/acl_y'], rec['imu/acl_z'],
            rec['imu/gyr_x'], rec['imu/gyr_y'], rec['imu/gyr_z'] ])
        return rec

    kl = KerasIMU()

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    '''
    def __init__(self, model=None, num_outputs=2, num_imu_inputs=6, input_shape=(120, 160, 3), *args, **kwargs):
        super(KerasIMU, self).__init__(*args, **kwargs)
        self.num_imu_inputs = num_imu_inputs
        self.model = default_imu(num_outputs = num_outputs, num_imu_inputs = num_imu_inputs, input_shape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr, accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z):
        # TODO: would be nice to take a vector input array.
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        imu_arr = np.array([accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z])\
                    .reshape(1, self.num_imu_inputs)
        outputs = self.model.predict([img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasBehavioral(KerasPilot):
    '''
    A Keras part that take an image and Behavior vector as input,
    outputs steering and throttle
    '''
    def __init__(self, model=None, num_outputs=2, num_behavior_inputs=2,
                 input_shape=(120, 160, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = default_bhv(num_outputs=num_outputs,
                                 num_bvh_inputs=num_behavior_inputs,
                                 input_shape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr, state_array):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        bhv_arr = np.array(state_array).reshape(1,len(state_array))
        angle_binned, throttle = self.model.predict([img_arr, bhv_arr])
        # in order to support older models with linear throttle,
        # we will test for shape of throttle to see if it's the newer
        # binned version.
        N = len(throttle[0])

        if N > 0:
            throttle = dk.utils.linear_unbin(throttle, N=N, offset=0.0, R=0.5)
        else:
            throttle = throttle[0][0]
        angle_unbinned = dk.utils.linear_unbin(angle_binned)
        return angle_unbinned, throttle


class KerasLocalizer(KerasPilot):
    '''
    A Keras part that take an image as input,
    outputs steering and throttle, and localisation category
    '''
    def __init__(self, model=None, num_locations=8, input_shape=(120, 160, 3),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = default_loc(num_locations=num_locations,
                                 input_shape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           metrics=['acc'], loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle, throttle, track_loc = self.model.predict([img_arr])
        loc = np.argmax(track_loc[0])
        return angle, throttle, loc


def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return new_height, input_shape[1], input_shape[2]


def default_categorical(input_shape=(120, 160, 3), roi_crop=(0, 0)):

    opt = keras.optimizers.Adam()
    drop = 0.2
    # we now expect that cropping done elsewhere. we will adjust our expeected
    # image size here:
    input_shape = adjust_input_shape(input_shape, roi_crop)
    # First layer, input layer, Shape comes from camera.py resolution, RGB
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window,
    # 2wx2h stride, relu activation
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu',
                      name="conv2d_1")(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)
    # 32 features, 5px5p kernel window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu',
                      name="conv2d_2")(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)
    if input_shape[0] > 32:
        # 64 features, 5px5p kernal window, 2wx2h stride, relu
        x = Convolution2D(64, (5,5), strides=(2,2), activation='relu',
                          name="conv2d_3")(x)
    else:
        # 64 features, 5px5p kernal window, 2wx2h stride, relu
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu',
                          name="conv2d_3")(x)
    if input_shape[0] > 64:
        # 64 features, 3px3p kernal window, 2wx2h stride, relu
        x = Convolution2D(64, (3,3), strides=(2,2), activation='relu',
                          name="conv2d_4")(x)
    elif input_shape[0] > 32:
        # 64 features, 3px3p kernal window, 2wx2h stride, relu
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu',
                          name="conv2d_4")(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)
    # 64 features, 3px3p kernal window, 1wx1h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu',
                      name="conv2d_5")(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)
    # Possibly add MaxPooling (will make it less sensitive to position in
    # image).  Camera angle fixed, so may not to be needed
    # Flatten to 1D (Fully connected)
    x = Flatten(name='flattened')(x)
    # Classify the data into 100 features, make all negatives 0
    x = Dense(100, activation='relu', name="fc_1")(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)
    # Classify the data into 50 features, make all negatives 0
    x = Dense(50, activation='relu', name="fc_2")(x)
    # Randomly drop out 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)
    # categorical output of the angle
    # Connect every input with every output and output 15 hidden units. Use
    # Softmax to give percentage. 15 categories and find best one based off
    # percentage 0.0-1.0
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    #continous output of throttle
    # Reduce to 1 number, Positive number only
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out],
                  name='KerasCategorical')
    return model


def default_n_linear(num_outputs, input_shape=(120, 160, 3), roi_crop=(0, 0)):

    drop = 0.1
    # we now expect that cropping done elsewhere. we will adjust our expeected
    # image size here:
    input_shape = adjust_input_shape(input_shape, roi_crop)

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu',
                      name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu',
                      name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear',
                             name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name="KerasLinear")
    return model


def linear_square_plus_cnn(x, size='S', is_seq=False):
    drop = 0.02
    # This makes the picture square in 1 steps (assuming 3x4 input) in all
    # following layers
    if size in ['S', 'M']:
        filters = [16, 32, 64, 96]
        kernels = [(7, 7), (5, 5), (3, 3), (2, 2)]
        if size == 'M':
            filters += [128]
            kernels += [(2, 2)]
    if size == 'L':
        filters = [20, 40, 80, 120, 160]
        kernels = [(9, 9), (7, 7), (5, 5), (3, 3), (2, 2)]
    if size == 'S':
        strides = [(3, 4), (2, 2)] + [(1, 1)] * 2
    else:  # M or L
        strides = [(3, 4)] + [(1, 1)] * 4

    # build CNN layers with data as above and batch norm, pooling & dropout
    for i, f, k, s in zip(range(len(filters)), filters, kernels, strides):
        conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                      activation='relu', name='conv' + str(i))
        norm = BatchNormalization(name='batch_norm' + str(i))
        pool = AveragePooling2D(pool_size=(2, 2), padding='same',
                                name='pool' + str(i))
        dropout = Dropout(rate=drop, name='drop' + str(i))
        if is_seq:
            x = TD(conv)(x)
            x = TD(norm)(x)
            x = TD(pool)(x)
            x = TD(dropout)(x)
        else:
            x = conv(x)
            x = norm(x)
            x = pool(x)
            x = dropout(x)

    flat = Flatten(name='flattened')
    x = TD(flat)(x) if is_seq else flat(x)
    return x


def square_plus_dense(size='S'):
    if size == 'S':
        layers = [96] * 4 + [48]
    elif size == 'M':
        layers = [128] * 5 + [64]
    elif size == 'L':
        layers = [144] * 8
    else:
        raise ValueError('size must be S, M or L but was', size)
    return layers


def linear_square_plus(input_shape=(120, 160, 3), roi_crop=(0, 0),
                       size='S', seq_len=None):

    # L2 regularisation
    l2 = 0.001
    input_shape = adjust_input_shape(input_shape, roi_crop)
    if seq_len:
        input_shape = (seq_len,) + input_shape
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = linear_square_plus_cnn(x, size, seq_len is not None)
    layers = square_plus_dense(size)
    for i, l in zip(range(len(layers)), layers):
        if seq_len:
            x = LSTM(units=l,
                     kernel_regularizer=regularizers.l2(l2),
                     name='lstm' + str(i),
                     return_sequences=(i != len(layers)-1))(x)
        else:
            x = Dense(units=l, activation='relu',
                      kernel_regularizer=regularizers.l2(l2),
                      name='dense' + str(i))(x)

    angle_out = Dense(units=1, activation='linear', name='angle')(x)
    throttle_out = Dense(units=1, activation='linear', name='throttle')(x)
    name = 'SquarePlus_' + size
    if seq_len:
        name += '_' + str(seq_len) + '_lstm'
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out], name=name)
    return model


def linear_square_plus_imu(input_shape=(120, 160, 3), roi_crop=(0, 0),
                           imu_dim=6, size='S'):
    assert 0 < imu_dim <= 6, 'imu_dim must be number in [1,..,6]'
    l2 = 0.001
    input_shape = adjust_input_shape(input_shape, roi_crop)
    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(imu_dim,), name="imu_in")
    x = img_in
    x = linear_square_plus_cnn(x, size)

    y = imu_in
    units = 24
    if size == 'M':
        units = 36
    elif size == 'L':
        units = 48
    y = Dense(units=units, activation='relu',
              kernel_regularizer=regularizers.l2(l2),
              name='dense_imu')(y)
    z = concatenate([x, y])
    layers = square_plus_dense(size)
    for i, l in zip(range(len(layers)), layers):
        z = Dense(units=l, activation='relu',
                  kernel_regularizer=regularizers.l2(l2),
                  name='dense' + str(i))(z)

    angle_out = Dense(units=1, activation='linear', name='angle')(z)
    throttle_out = Dense(units=1, activation='linear', name='throttle')(z)
    model = Model(inputs=[img_in, imu_in], outputs=[angle_out, throttle_out],
                  name='SquarePlusImu_' + size + '_' + str(imu_dim))
    return model


def default_imu(num_outputs, num_imu_inputs, input_shape):
    # We now expect that cropping done elsewhere. we will adjust our expected
    # image size here: input_shape = adjust_input_shape(input_shape, roi_crop)

    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)

    y = imu_in
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []

    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))

    model = Model(inputs=[img_in, imu_in], outputs=outputs, name="KerasIMU")

    return model


def default_bhv(num_outputs, num_bvh_inputs, input_shape):
    '''
    Notes: this model depends on concatenate which failed on keras < 2.0.8
    '''

    img_in = Input(shape=input_shape, name='img_in')
    bvh_in = Input(shape=(num_bvh_inputs,), name="behavior_in")

    x = img_in
    # x = Cropping2D(cropping=((60,0), (0,0)))(x) #trim 60 pixels off top
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)

    y = bvh_in
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)

    z = concatenate([x, y])
    z = Dense(100, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    # categorical output of the angle
    # Connect every input with every output and output 15 hidden units. Use
    # Softmax to give percentage. 15 categories and find best one based off
    # percentage 0.0-1.0
    angle_out = Dense(15, activation='softmax', name='angle_out')(z)
    # continous output of throttle
    # Reduce to 1 number, Positive number only
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(z)
    model = Model(inputs=[img_in, bvh_in], outputs=[angle_out, throttle_out],
                  name='KerasBehavioural')
    return model


def default_loc(num_locations, input_shape):
    drop = 0.2

    img_in = Input(shape=input_shape, name='img_in')

    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu',
                      name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu',
                      name="conv2d_5")(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)

    z = Dense(50, activation='relu')(x)
    z = Dropout(drop)(z)

    # linear output of the angle
    angle_out = Dense(1, activation='linear', name='angle')(z)

    # linear output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle')(z)

    # categorical output of location
    loc_out = Dense(num_locations, activation='softmax', name='loc')(z)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out, loc_out],
                  name='KerasLocalizer')

    return model


class KerasRNN_LSTM(KerasPilot):
    def __init__(self, image_w=160, image_h=120, image_d=3, seq_length=3,
                 num_outputs=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        image_shape = (image_h, image_w, image_d)
        self.model = rnn_lstm(seq_length=seq_length,
                              num_outputs=num_outputs,
                              image_shape=image_shape)
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.compile()
        self.optimizer = "rmsprop"

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)

        img_arr = np.array(self.img_seq).reshape(1, self.seq_length,
                                                 self.image_h, self.image_w,
                                                 self.image_d)
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle


def rnn_lstm(seq_length=3, num_outputs=2, image_shape=(120, 160, 3)):
    # we now expect that cropping done elsewhere. we will adjust our
    # expected image size here:
    # input_shape = adjust_input_shape(input_shape, roi_crop)

    img_seq_shape = (seq_length,) + image_shape
    img_in = Input(batch_shape=img_seq_shape, name='img_in')
    drop_out = 0.3

    x = Sequential(name='KerasLSTM')
    x.add(TD(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'),
             input_shape=img_seq_shape))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (3, 3), strides=(1, 1), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(MaxPooling2D(pool_size=(2, 2))))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(drop_out)))

    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_fin"))
    x.add(Dropout(.1))
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(64, activation='relu'))
    x.add(Dense(10, activation='relu'))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))

    return x


class Keras3D_CNN(KerasPilot):
    def __init__(self, image_w=160, image_h=120, image_d=3, seq_length=20,
                 num_outputs=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = build_3d_cnn(w=image_w, h=image_h, d=image_d, s=seq_length,
                                  num_outputs=num_outputs)
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.compile()

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer,
                           metrics=['accuracy'])

    def run(self, img_arr):

        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)

        img_arr = np.array(self.img_seq).reshape(1, self.seq_length,
                                                 self.image_h, self.image_w,
                                                 self.image_d)
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle


def build_3d_cnn(w, h, d, s, num_outputs):
    # Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model
    # /model.py
    '''
        w : width
        h : height
        d : depth
        s : n_stacked
    '''
    input_shape=(s, h, w, d)

    model = Sequential(name='Keras3dCnn')
    # First layer
    # model.add(Cropping3D(cropping=((0,0), (50,10), (0,0)),
    # input_shape=input_shape) ) #trim pixels off top

    # Second layer
    model.add(Conv3D(
        filters=16, kernel_size=(3,3,3), strides=(1,3,3),
        data_format='channels_last', padding='same', input_shape=input_shape)
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Third layer
    model.add(Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fourth layer
    model.add(Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fifth layer
    model.add(Conv3D(
        filters=128, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fully connected layer
    model.add(Flatten(name='flattened'))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    #model.add(Activation('tanh'))

    return model


class KerasLatent(KerasPilot):
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = default_latent(num_outputs, input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss={
            "img_out" : "mse", "n_outputs0" : "mse", "n_outputs1" : "mse"
        }, loss_weights={
            "img_out" : 100.0, "n_outputs0" : 2.0, "n_outputs1" : 1.0
        })

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[1]
        throttle = outputs[2]
        return steering[0][0], throttle[0][0]


def default_latent(num_outputs, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu',
                      name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (3, 3), strides=(1, 1), activation='relu',
                      name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (3, 3), strides=(1, 1), activation='relu',
                      name="conv2d_5")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu',
                      name="conv2d_6")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu',
                      name="conv2d_7")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (1, 1), strides=(2, 2), activation='relu',
                      name="latent")(x)

    y = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2,
                        name="deconv2d_1")(x)
    y = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2,
                        name="deconv2d_2")(y)
    y = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2,
                        name="deconv2d_3")(y)
    y = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2,
                        name="deconv2d_4")(y)
    y = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2,
                        name="deconv2d_5")(y)
    y = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2,
                        name="img_out")(y)

    x = Flatten(name='flattened')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = [y]
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear',
                             name='n_outputs' + str(i))(x))
    model = Model(inputs=[img_in], outputs=outputs, name='KerasLatent')
    return model


class ModelLoader:
    """ This donkey part is meant to be continuously looking for an updated
    pilot to overwrite the existing one"""

    def __init__(self, keras_pilot, model_path):
        assert isinstance(keras_pilot, KerasPilot) \
            or isinstance(keras_pilot, TFLitePilot), \
            'ModelLoader only works with KerasPilot or TFLitePilot'
        self.remote_model = keras_pilot
        self.model_path = model_path
        self.update_trigger = False
        self.is_updated = False
        # make own copy of the model
        self.model = copy.copy(keras_pilot)

    def run_threaded(self, update):
        """
        Donkey part interface for threaded parts
        :param bool update:
            Should be true if the model file changed otherwise false. The
            input here is expected to come from the output of the FileWatcher.
        :return bool:
            Indicator if the remote model was updated
        """
        if update:
            # if FileWatcher recognises a new version of the file set the flag
            self.update_trigger = True
            return False
        else:
            # check pilot is loaded and we shove it into the remote object
            if self.is_updated:
                self.remote_model.update(self.model)
                # reset update state to false so we don't update until
                # the flag gets set to true by the file watcher again
                self.is_updated = False
                print('ModelLoader updated model.')
                return True
            # otherwise no updated model available - do nothing
            else:
                return False

    def update(self):
        """
        Donkey parts interface
        """
        while True:
            # self.update has been set by the FileWatcher in the loop
            if self.update_trigger:
                self.model.load(model_path=self.model_path)
                self.update_trigger = False
                self.is_updated = True
            else:
                # no update wait 1s
                time.sleep(1.0)

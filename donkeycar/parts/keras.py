"""

keras.py

Methods to create, use, save and load pilots. Pilots contain the highlevel
logic used to determine the angle and throttle of a vehicle. Pilots can
include one or more models to help direct the vehicles motion.

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

import donkeycar as dk
from donkeycar.utils import normalize_image, linear_bin
from donkeycar.pipeline.types import TubRecord
from donkeycar.parts.interpreter import Interpreter, KerasInterpreter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2DTranspose
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

ONE_BYTE_SCALE = 1.0 / 255.0

# type of x
XY = Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]


logger = getLogger(__name__)


class KerasPilot(ABC):
    """
    Base class for Keras models that will provide steering and throttle to
    guide a car.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        # self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        logger.info(f'Created {self} with interpreter: {interpreter}')

    def load(self, model_path: str) -> None:
        logger.info(f'Loading model {model_path}')
        self.interpreter.load(model_path)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception(f"Unknown optimizer type: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    def get_input_shapes(self) -> List[tf.TensorShape]:
        return self.interpreter.get_input_shapes()

    def seq_size(self) -> int:
        return 0

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        norm_arr = normalize_image(img_arr)
        np_other_array = np.array(other_arr) if other_arr else None
        return self.inference(norm_arr, np_other_array)

    def inference(self, img_arr: np.ndarray, other_arr: Optional[np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Inferencing using the interpreter
            :param img_arr:     float32 [0,1] numpy array with normalized image
                                data
            :param other_arr:   numpy array of additional data to be used in the
                                pilot, like IMU array for the IMU model or a
                                state vector in the Behavioural model
            :return:            tuple of (angle, throttle)
        """
        out = self.interpreter.predict(img_arr, other_arr)
        return self.interpreter_to_output(out)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Inferencing using the interpreter
            :param input_dict:  input dictionary of str and np.ndarray
            :return:            typically tuple of (angle, throttle)
        """
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(
            self,
            interpreter_out: Sequence[Union[float, np.ndarray]]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Virtual method to be implemented by child classes for conversion
            :param interpreter_out:  input data
            :return:                 output values, possibly tuple of np.ndarray
        """
        pass

    def train(self,
              model_path: str,
              train_data: Union[DatasetV1, DatasetV2],
              train_steps: int,
              batch_size: int,
              validation_data: Union[DatasetV1, DatasetV2],
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5,
              show_plot: bool = False) -> tf.keras.callbacks.History:
        """
        trains the model
        """
        assert isinstance(self.interpreter, KerasInterpreter)
        model = self.interpreter.model
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]

        history: tf.keras.callbacks.History = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False)

        if show_plot:
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path

                plt.figure(1)
                # Only do accuracy if we have that data
                # (e.g. categorical outputs)
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

                plt.savefig(Path(model_path).with_suffix('.png'))
                # plt.show()

            except Exception as ex:
                print(f"problems with loss graph: {ex}")

        return history

    def x_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Return x from record, default returns only image array"""
        assert isinstance(record, TubRecord), "TubRecord required"
        img_arr = record.image(cached=True)
        return img_arr

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        """ Translates x into dictionary where all model input layer's names
            must be matched by dictionary keys. """
        return {'img_in': x}

    def x_transform_and_process(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) -> XY:
        """ Transforms the record into x for training the model to x,y, and
            applies an image augmentation. Here we assume the model only takes
            the image as input. """
        x_img = self.x_transform(record)
        # apply augmentation / normalisation
        x_process = img_processor(x_img)
        return x_process

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Return y from record, needs to be implemented"""
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        """ Translates y into dictionary where all model output layer's names
            must be matched by dictionary keys. """
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self) -> Tuple[Dict[str, np.typename], ...]:
        """ Used in tf.data, assume all types are doubles"""
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self) -> Dict[str, tf.TensorShape]:
        return {}

    def __str__(self) -> str:
        """ For printing model initialisation """
        return type(self).__name__

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


class KerasCategorical(KerasPilot):
    """
    The KerasCategorical pilot breaks the steering and throttle decisions
    into discreet angles and then uses categorical cross entropy to train the
    network to activate a single neuron for each steering and throttle
    choice. This can be interesting because we get the confidence value as a
    distribution over all choices. This uses the dk.utils.linear_bin and
    dk.utils.linear_unbin to transform continuous real numbers into a range
    of discreet values for training and runtime. The input and output are
    therefore bounded and must be chosen wisely to match the data. The
    default ranges work for the default setup. But cars which go faster may
    want to enable a higher throttle range. And cars with larger steering
    throw may want more bins.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 throttle_range: float = 0.5):
        super().__init__(interpreter, input_shape)
        self.throttle_range = throttle_range

    def create_model(self):
        return default_categorical(self.input_shape)

    def compile(self):
        self.interpreter.compile(
            optimizer=self.optimizer,
            metrics=['accuracy'],
            loss={'angle_out': 'categorical_crossentropy',
                  'throttle_out': 'categorical_crossentropy'},
            loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})

    def interpreter_to_output(self, interpreter_out):
        angle_binned, throttle_binned = interpreter_out
        N = len(throttle_binned)
        throttle = dk.utils.linear_unbin(throttle_binned, N=N,
                                         offset=0.0, R=self.throttle_range)
        angle = dk.utils.linear_unbin(angle_binned)
        return angle, throttle

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        angle = linear_bin(angle, N=15, offset=1, R=2.0)
        throttle = linear_bin(throttle, N=20, offset=0.0, R=self.throttle_range)
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, tuple), 'Expected tuple'
        angle, throttle = y
        return {'angle_out': angle, 'throttle_out': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle_out': tf.TensorShape([15]),
                   'throttle_out': tf.TensorShape([20])})
        return shapes


class KerasLinear(KerasPilot):
    """
    The KerasLinear pilot uses one neuron to output a continuous value via
    the Keras Dense layer with linear activation. One each for steering and
    throttle. The output is not bounded.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, tuple), 'Expected tuple'
        angle, throttle = y
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes


class KerasMemory(KerasLinear):
    """
    The KerasLinearWithMemory is based on KerasLinear but uses the last n
    steering and throttle commands as input in order to produce smoother
    steering outputs
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 mem_length: int = 3,
                 mem_depth: int = 0):
        self.mem_length = mem_length
        self.mem_seq: List[np.array] = []
        self.mem_depth = mem_depth
        super().__init__(interpreter, input_shape)

    def seq_size(self) -> int:
        return self.mem_length + 1

    def create_model(self):
        return default_memory(self.input_shape,
                              self.mem_length, self.mem_depth, )

    def load(self, model_path: str) -> None:
        super().load(model_path)
        self.mem_length = self.interpreter.get_input_shapes()[1][1] // 2
        logger.info(f'Loaded mem length {self.mem_length}')

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) -> \
            Tuple[Union[float, np.ndarray], ...]:
        while len(self.mem_seq) < self.mem_length:
            self.mem_seq.append(other_arr)

        self.mem_seq = self.mem_seq[1:]
        self.mem_seq.append(other_arr)
        np_mem_arr = np.array(self.mem_seq).reshape((2 * self.mem_length,))
        img_arr_norm = normalize_image(img_arr)
        return super().inference(img_arr_norm, np_mem_arr)

    def x_transform(self, records: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Return x from record, here x = image, previous angle/throttle
            values """
        assert isinstance(records, list), 'List[TubRecord] expected'
        assert len(records) == self.mem_length + 1, \
            f"Record list of length {self.mem_length} required but " \
            f"{len(records)} was passed"
        img_arr = records[-1].image(cached=True)
        mem = [[r.underlying['user/angle'], r.underlying['user/throttle']]
               for r in records[:-1]]
        return img_arr, np.array(mem).reshape((2 * self.mem_length,))

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        """ Translates x into dictionary where all model input layer's names
            must be matched by dictionary keys. """
        assert(isinstance(x, tuple)), 'Tuple expected'
        img_arr, mem = x
        return {'img_in': img_arr, 'mem_in': mem}

    def x_transform_and_process(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) -> XY:
        """ Transforms the record into x for training the model to x,y,
            here we assume the model only takes the image as input. """
        xt = self.x_transform(record)
        assert isinstance(xt, tuple), 'Tuple expected'
        x_img, mem = xt
        # apply augmentation / normalisation
        x_process = img_processor(x_img)
        return x_process, mem

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return angle, throttle

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'mem_in': tf.TensorShape(2 * self.mem_length)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return super().__str__() \
            + f'-L:{self.mem_length}-D:{self.mem_depth}'


class KerasInferred(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)):
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(1, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        return steering, dk.utils.throttle(steering)

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        return angle

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, float), 'Float expected'
        return {'n_outputs0': y}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([])})
        return shapes

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
        print(self.text())
        self.compile()

    def text(self):
        return 'Created ' + self.__class__.__name__ + ' NN size: ' \
                   + str(self.size)

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

    def text(self):
        return super().text() + ' Seq len: ' + str(self.seq_length)

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
    def __init__(self, input_shape=(120, 160, 3), roi_crop=(0, 0),
                 *args, **kwargs):
        self.imu_dim = kwargs.get('imu_dim', 6)
        super().__init__(input_shape, roi_crop, *args, **kwargs)

    def make_model(self, input_shape, roi_crop):
        model = linear_square_plus_imu(input_shape, roi_crop,
                                       imu_dim=self.imu_dim, size=self.size)
        return model

    def text(self):
        return super().text() + ' Imu dim ' + str(self.imu_dim)

    def run(self, img_arr, imu_arr=None):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        if imu_arr is None:
            imu_arr = np.zeros((1, 6))
        else:
            imu_arr = np.array(imu_arr).reshape(1, self.imu_dim)
        outputs = self.model.predict(x=[img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasSquarePlusImuLstm(KerasSquarePlusLstm):
    def __init__(self, input_shape=(120, 160, 3), roi_crop=(0, 0),
                 *args, **kwargs):
        self.imu_dim = kwargs.get('imu_dim', 6)
        self.imu_seq = []
        super().__init__(input_shape=input_shape, roi_crop=roi_crop,
                         *args, **kwargs)

    def make_model(self, input_shape, roi_crop):
        return linear_square_plus_imu(input_shape, roi_crop,
                                      imu_dim=self.imu_dim,
                                      size=self.size,
                                      seq_len=self.seq_length)

    def run(self, img_arr, imu_arr):
        # convert imu python list into numpy array first, img_arr is already
        # numpy array
        imu_arr = np.array(imu_arr)
        # if buffer empty fill to length
        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)
            self.imu_seq.append(imu_arr)
        # pop oldest img / imu from front and append current img / imu at end
        self.img_seq.pop(0)
        self.img_seq.append(img_arr)
        self.imu_seq.pop(0)
        self.imu_seq.append(imu_arr)
        # reshape and run model
        new_img_shape = (1, self.seq_length, ) + img_arr.shape
        img_arr = np.array(self.img_seq).reshape(new_img_shape)
        new_imu_shape = (1, self.seq_length, ) + imu_arr.shape
        imu_arr = np.array(self.imu_seq).reshape(new_imu_shape)
        outputs = self.model.predict([img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasWorld(KerasSquarePlus):
    """
    World model for pilot. Allows pre-trained model from which it only takes
    the encoder that transforms images into latent vectors.
    """

    def __init__(self, input_shape=(144, 192, 3), roi_crop=(0, 0),
                 encoder_path=None, *args, **kwargs):
        self.encoder_path = encoder_path
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.encoder = self.make_encoder(input_shape)
        self.encoder.trainable = self.encoder_path is None
        super().__init__(input_shape, roi_crop, size='R', *args, **kwargs)

    def make_encoder(self, input_shape):
        if self.encoder_path is None:
            return AutoEncoder(input_shape=input_shape,
                               latent_dim=self.latent_dim).encoder
        else:
            # load full world model
            k_world = tf.keras.models.load_model(self.encoder_path)
            # return layer 1 which is the encoder
            encoder = k_world.layers[1]
            self.encoder_checks(encoder)
            # reset latent dimension from loaded model
            self.latent_dim = encoder.outputs[0].shape[1]
            return encoder

    def make_controller_inputs(self):
        latent_input = keras.Input(shape=(self.latent_dim,), name='latent_in')
        return latent_input

    def transform_controller_inputs(self, inputs):
        return inputs

    def make_controller(self):
        l2 = 0.001
        inputs = self.make_controller_inputs()
        x = self.transform_controller_inputs(inputs)
        for i in range(3):
            x = Dense(units=self.latent_dim / 2, activation='relu',
                      kernel_regularizer=regularizers.l2(l2),
                      name='dense' + str(i))(x)
        angle_out = Dense(units=1, activation='linear', name='angle')(x)
        throttle_out = Dense(units=1, activation='linear', name='throttle')(x)
        controller = Model(inputs=inputs,
                           outputs=[angle_out, throttle_out],
                           name='controller')
        return controller

    def make_model(self, input_shape, roi_crop):
        controller = self.make_controller()
        pilot_input = keras.Input(shape=input_shape)
        encoded_img = self.encoder(pilot_input)
        [angle, throttle] = controller(encoded_img)
        model = Model(pilot_input, [angle, throttle], name="world_pilot")
        return model

    def text(self):
        text = super().text()
        if self.encoder_path is not None:
            text += ' with encoder from : ' + str(self.encoder_path)
        return text

    def load_encoder(self, model_path):
        print('Loading encoder from', model_path, 'into world model...', end='')
        model = keras.models.load_model(model_path)
        encoder = model.layers[1]
        self.encoder_checks(encoder)
        self.encoder = encoder
        self.latent_dim = self.encoder.outputs[0].shape[1]
        self.encoder.trainable = False
        input_shape = encoder.inputs[0].shape[1:]
        self.model = self.make_model(input_shape=input_shape, roi_crop=(0, 0))
        print("done - encoder is not trainable now")

    def encoder_checks(self, encoder):
        assert type(encoder) is tf.keras.Model, \
            'first layer of model needs to be a model'
        assert encoder.name == 'encoder', \
            'first layer model should have name "encoder"'


class KerasWorldImu(KerasWorld, KerasSquarePlusImu):
    """
    World model for pilot. Uses pre-trained encoder which breaks down images
    into latent vectors.
    """

    def __init__(self, input_shape=(144, 192, 3), roi_crop=(0, 0),
                 encoder_path=None, *args, **kwargs):
        super().__init__(input_shape, roi_crop,
                         encoder_path=encoder_path, *args, **kwargs)

    def make_controller_inputs(self):
        latent_input = keras.Input(shape=(self.latent_dim,), name='latent_in')
        imu_input = keras.Input(shape=(self.imu_dim,), name='imu_in')
        return [latent_input, imu_input]

    def transform_controller_inputs(self, inputs):
        return concatenate(inputs)

    def make_model(self, input_shape, roi_crop):
        controller = self.make_controller()
        pilot_input = keras.Input(shape=input_shape, name='img_in')
        encoded_img = self.encoder(pilot_input)
        imu_input = keras.Input(shape=(self.imu_dim,), name='imu_in')
        [angle, throttle] = controller([encoded_img, imu_input])
        model = Model(inputs=[pilot_input, imu_input],
                      outputs=[angle, throttle],
                      name="world_pilot_imu")
        return model


class WorldMemory:
    def __init__(self, encoder_path='models/encoder.h5', *args, **kwargs):
        self.seq_length = kwargs.get('seq_length', 3)
        #self.imu_dim = kwargs.get('imu_dim', 6)
        self.layers = kwargs.get('lstm_layers', 3)
        self.units = kwargs.get('lstm_units', 128)
        self.encoder = keras.models.load_model(encoder_path).layers[1]
        self.latent_dim = self.encoder.outputs[0].shape[1]
        self.model = self.make_model()
        print('Created WorldMemory with encoder path:', encoder_path,
              'seq length', self.seq_length)

    def make_model(self):
        l2 = 0.001
        input_shape_latent = (self.seq_length, self.latent_dim)
        #input_shape_imu = (self.seq_length, self.imu_dim)
        input_shape_drive = (self.seq_length, 2)
        latent_seq_in = keras.Input(input_shape_latent, name='latent_seq_in')
        #imu_seq_in = keras.Input(input_shape_imu, name='imu_seq_in')
        drive_seq_in = keras.Input(input_shape_drive, name='drive_seq_in')
        inputs = [latent_seq_in, drive_seq_in]
        x = concatenate(inputs, axis=-1)
        for i in range(self.layers):
            last = (i == self.layers - 1)
            x = LSTM(units=self.units,
                     kernel_regularizer=regularizers.l2(l2),
                     recurrent_regularizer=regularizers.l2(l2),
                     name='lstm' + str(i),
                     return_sequences=not last,
                     return_state=last)(x)
        # now x[0] has return sequences and x[1] has the state
        sequences = x[0]
        state = x[1]
        latent_out = Dense(units=self.latent_dim, name='latent_out')(sequences)
        #imu_out = Dense(units=self.imu_dim, name='imu_out')(sequences)
        #drive_out = Dense(units=2, name='drive_out')(sequences)
        outputs = [latent_out, state]  # [latent_out, imu_out, drive_out, state]
        model = Model(inputs=inputs, outputs=outputs, name='Memory')
        return model

    def load(self, model_path):
        prev = self.model_id()
        self.model = keras.models.load_model(model_path)
        print("Load model: overwriting " + prev + " with " + self.model_id())

    def model_id(self):
        return 'world_memory layers: ' + str(self.layers) + ' units: ' \
               + str(self.units)

    def compile(self):
        # here we set the loss for the internal state output to None so it
        # doesn't get used in training
        opt = tf.keras.optimizers.Adam(learning_rate=0.01,
                                       beta_1=0.9,
                                       beta_2=0.999)
        self.model.compile(optimizer=opt, loss=['mse', None])


class WorldPilot(KerasWorldImu):
    def __init__(self, encoder_path=None, memory_path=None,
                 input_shape=(144, 192, 3), roi_crop=(0, 0), *args, **kwargs):
        if memory_path:
            assert encoder_path, "Need to provide encoder path too"
            self.memory = keras.models.load_model(memory_path)
            # get seq length from input shape of memory model
            self.seq_length = self.memory.inputs[0].shape[1]
            # get state vector length from last output of memory model
            self.state_dim = self.memory.outputs[1].shape[1]
            super().__init__(input_shape=input_shape, roi_crop=roi_crop,
                             encoder_path=encoder_path)
            # fill sequence to size
            self.latent_seq = [np.zeros((self.latent_dim,))] * self.seq_length
            self.drive_seq = [np.zeros((2,))] * self.seq_length
            self.encoder.trainable = False
            self.memory.trainable = False
        else:
            assert encoder_path is None, "Don't pass encoder w/o memory model"
            self.memory = self.seq_length = self.state_dim = None
            self.latent_seq = self.drive_seq = None
            print('Created empty WorldPilot - you need to load a model')

    def make_controller_inputs(self):
        latent_input = keras.Input(shape=(self.latent_dim,), name='latent_in')
        imu_input = keras.Input(shape=(self.imu_dim,), name='imu_in')
        state_input = keras.Input(shape=(self.state_dim,), name='state_in')
        return [latent_input, imu_input, state_input]

    def make_model(self, input_shape, roi_crop):
        latent_seq_input = keras.Input(shape=(self.seq_length,
                                              self.latent_dim),
                                       name='latent_seq_in')
        drive_seq_input = keras.Input(shape=(self.seq_length, 2),
                                      name='drive_seq_in')
        [latent_out, state] = self.memory([latent_seq_input, drive_seq_input])
        img_input = keras.Input(shape=input_shape, name='img_in')
        latent = self.encoder(img_input)
        imu_input = keras.Input(shape=(self.imu_dim,), name='imu_in')
        controller = self.make_controller()
        [angle, throttle] = controller([latent, imu_input, state])
        model = Model(inputs=[img_input, imu_input,
                              latent_seq_input, drive_seq_input],
                      outputs=[angle, throttle, latent],
                      name='world_pilot')
        return model

    def compile(self):
        # here we set the loss for the latent vector output to None so it
        # doesn't get used in training
        self.model.compile(optimizer='adam', loss=['mse', 'mse', None])

    def model_id(self):
        return 'WorldPilot_sd_{:}_ld_{:}_seql_{:}'\
            .format(self.state_dim, self.latent_dim, self.seq_length)

    def load(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(self.model.summary())
        self.memory = self.model.get_layer('Memory')
        # get seq length from input shape of memory model
        self.seq_length = self.memory.inputs[0].shape[1]
        # get state vector length from last output of memory model
        self.state_dim = self.memory.outputs[1].shape[1]
        self.encoder = self.model.get_layer('encoder')
        self.latent_dim = self.encoder.outputs[0].shape[1]
        self.imu_dim = self.model.get_layer('imu_in').output_shape[0][1]
        # fill sequence to size
        self.latent_seq = [np.zeros((self.latent_dim,))] * self.seq_length
        self.drive_seq = [np.zeros((2,))] * self.seq_length
        self.encoder.trainable = False
        self.memory.trainable = False

    def run(self, img_arr, imu_arr=None):
        # convert imu python list into numpy array first, img_arr is already
        # numpy array
        if imu_arr is None:
            imu_arr = np.zeros((1, self.imu_dim))
        else:
            imu_arr = np.array([imu_arr])
        # convert image array into latent vector first
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # now we run the model returning drive vector and last latent vector
        inputs = [img_arr, imu_arr,
                  np.array([self.latent_seq]), np.array([self.drive_seq])]
        [angle, throttle, latent] = self.model.predict(inputs)
        # convert drive array of ndarray into more convenient type
        drive_arr = np.array([angle[0][0], throttle[0][0]])
        # add new results for angle, throttle and latent vector into sequence
        for seq, new in zip([self.latent_seq, self.drive_seq],
                            [latent[0], drive_arr]):
            # pop oldest entry from front and append new entry at end
            seq.pop(0)
            seq.append(new)
        # return angle, throttle
        return drive_arr[0], drive_arr[1]


class AutoEncoder:
    def __init__(self, input_shape=(144, 192, 3), latent_dim=256,
                 encoder_path=None, decoder_path=None):
        self.input_shape = input_shape
        self.output_shape = None
        self.latent_dim = latent_dim
        self.filters = [16, 32, 48, 64, 80]
        self.kernel = (3, 3)
        self.strides = [(2, 2)] + [(1, 1)] * 5
        self.encoder = keras.models.load_model(encoder_path) if encoder_path \
            else self.make_encoder()
        self.decoder = keras.models.load_model(decoder_path) if decoder_path \
            else self.make_decoder()
        img_input = keras.Input(shape=(144, 192, 3))
        encoded_img = self.encoder(img_input)
        decoded_img = self.decoder(encoded_img)
        self.autoencoder = keras.Model(img_input, decoded_img,
                                       name="autoencoder")

    def make_encoder(self):
        encoder_input = keras.Input(self.input_shape, name='img_in')
        x = encoder_input
        drop = 0.02
        conv = None
        num_l = len(self.filters)
        for i, f, s in zip(range(num_l), self.filters, self.strides):
            conv = Conv2D(filters=f, kernel_size=self.kernel, strides=s,
                          padding='same', activation='relu',
                          name='conv' + str(i))
            x = conv(x)
            x = BatchNormalization(name='batch_norm' + str(i))(x)
            x = Dropout(rate=drop, name='drop' + str(i))(x)
            if i < num_l - 1:
                x = MaxPooling2D(pool_size=(3, 3) if i == num_l - 2 else (2, 2),
                                 padding='same',
                                 name='pool' + str(i))(x)

        # remove first entry from (, a, b, c)
        self.output_shape = tuple(conv.output_shape[1:])
        x = Flatten()(x)
        latent = Dense(self.latent_dim, name="dense", activation='sigmoid')(x)
        encoder = keras.Model(encoder_input, latent, name="encoder")
        return encoder

    def make_decoder(self):
        latent_input = keras.Input(shape=(self.latent_dim,), name='latent_in')
        dim = np.prod(self.output_shape)
        x = Dense(dim, activation="relu")(latent_input)
        x = Reshape(self.output_shape)(x)
        for i, f, s in zip(reversed(range(len(self.filters))),
                           reversed(self.filters), reversed(self.strides)):
            if i < 4:
                x = UpSampling2D((3, 3) if i == 3 else (2, 2))(x)
            x = Conv2DTranspose(f, self.kernel, activation="relu",
                                strides=s, padding="same",
                                name='deconv' + str(i))(x)
        decoder_output = Conv2DTranspose(3, self.kernel,
                                         strides=self.strides[0],
                                         activation="sigmoid",
                                         padding="same",
                                         name='deconv_convert')(x)
        decoder = keras.Model(latent_input, decoder_output, name="decoder")
        return decoder


class KerasIMU(KerasPilot):
    """
    A Keras part that take an image and IMU vector as input,
    outputs steering and throttle

    Note: When training, you will need to vectorize the input from the IMU.
    Depending on the names you use for imu records, something like this will
    work:

    X_keys = ['cam/image_array','imu_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(rec):
        rec['imu_array'] =
            np.array([ rec['imu/acl_x'], rec['imu/acl_y'], rec['imu/acl_z'],
                       rec['imu/gyr_x'], rec['imu/gyr_y'], rec['imu/gyr_z'] ])
        return rec

    kl = KerasIMU()

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                    record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    """
    # keys for imu data in TubRecord
    imu_vec = [f'imu/{f}_{x}' for f in ('acl', 'gyr') for x in 'xyz']

    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2, num_imu_inputs: int = 6):
        self.num_outputs = num_outputs
        self.num_imu_inputs = num_imu_inputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_imu(num_outputs=self.num_outputs,
                           num_imu_inputs=self.num_imu_inputs,
                           input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def x_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        img_arr = record.image(cached=True)
        imu_arr = [record.underlying[k] for k in self.imu_vec]
        return img_arr, np.array(imu_arr)

    def x_transform_and_process(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) -> XY:
        # this transforms the record into x for training the model to x,y
        xt = self.x_transform(record)
        assert isinstance(xt, tuple), 'Tuple expected'
        x_img, x_imu = xt
        # here the image is in first slot of the tuple
        x_img_process = img_processor(x_img)
        return x_img_process, x_imu

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(x, tuple), 'Tuple required'
        return {'img_in': x[0], 'imu_in': x[1]}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, tuple), 'Expected tuple'
        angle, throttle = y
        return {'out_0': angle, 'out_1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'imu_in': tf.TensorShape([self.num_imu_inputs])},
                  {'out_0': tf.TensorShape([]),
                   'out_1': tf.TensorShape([])})
        return shapes


class KerasBehavioral(KerasCategorical):
    """
    A Keras part that take an image and Behavior vector as input,
    outputs steering and throttle
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 throttle_range: float = 0.5,
                 num_behavior_inputs: int = 2):
        self.num_behavior_inputs = num_behavior_inputs
        super().__init__(interpreter, input_shape, throttle_range)

    def create_model(self):
        return default_bhv(num_bvh_inputs=self.num_behavior_inputs,
                           input_shape=self.input_shape)

    def x_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        img_arr = record.image(cached=True)
        bhv_arr = record.underlying['behavior/one_hot_state_array']
        return img_arr, np.array(bhv_arr)

    def x_transform_and_process(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) -> XY:
        # this transforms the record into x for training the model to x,y
        xt = self.x_transform(record)
        assert isinstance(xt, tuple), 'Tuple expected'
        x_img, bhv_arr = xt
        # here the image is in first slot of the tuple
        x_img_process = img_processor(x_img)
        return x_img_process, bhv_arr

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(x, tuple), 'Tuple required'
        return {'img_in': x[0], 'xbehavior_in': x[1]}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'xbehavior_in': tf.TensorShape([self.num_behavior_inputs])},
                  {'angle_out': tf.TensorShape([15]),
                   'throttle_out': tf.TensorShape([20])})
        return shapes


class KerasLocalizer(KerasPilot):
    """
    A Keras part that take an image as input,
    outputs steering and throttle, and localisation category
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_locations: int = 8):
        self.num_locations = num_locations
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_loc(num_locations=self.num_locations,
                           input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, metrics=['acc'],
                                 loss='mse')

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        angle, throttle, track_loc = interpreter_out
        loc = np.argmax(track_loc)
        return angle[0], throttle[0], loc

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) -> XY:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        loc = record.underlying['localizer/location']
        loc_one_hot = np.zeros(self.num_locations)
        loc_one_hot[loc] = 1
        return angle, throttle, loc_one_hot

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, tuple), 'Expected tuple'
        angle, throttle, loc = y
        return {'angle': angle, 'throttle': throttle, 'zloc': loc}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle': tf.TensorShape([]),
                   'throttle': tf.TensorShape([]),
                   'zloc': tf.TensorShape([self.num_locations])})
        return shapes


class KerasLSTM(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 seq_length=3,
                 num_outputs=2):
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        super().__init__(interpreter, input_shape)
        self.img_seq: List[np.ndarray] = []
        self.optimizer = "rmsprop"

    def seq_size(self) -> int:
        return self.seq_length

    def create_model(self):
        return rnn_lstm(seq_length=self.seq_length,
                        num_outputs=self.num_outputs,
                        input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def x_transform(self, records: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Return x from record, here x = stacked images """
        assert isinstance(records, list), 'List[TubRecord] expected'
        assert len(records) == self.seq_length, \
            f"Record list of length {self.seq_length} required but " \
            f"{len(records)} was passed"
        img_arrays = [rec.image(cached=True) for rec in records]
        return np.array(img_arrays)

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        """ Translates x into dictionary where all model input layer's names
            must be matched by dictionary keys. """
        img_arr = x
        return {'img_in': img_arr}

    def x_transform_and_process(
            self,
            records: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) -> XY:
        """ Transforms the record sequence into x for training the model to
            x, y. """
        img_seq = self.x_transform(records)
        assert isinstance(img_seq, np.ndarray)
        # apply augmentation / normalisation on sequence of images
        x_process = [img_processor(img) for img in img_seq]
        return np.array(x_process)

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Only return the last entry of angle/throttle"""
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, tuple), 'Expected tuple'
        return {'model_outputs': list(y)}

    def run(self, img_arr, other_arr=None):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)
        new_shape = (self.seq_length, *self.input_shape)
        img_arr = np.array(self.img_seq).reshape(new_shape)
        img_arr_norm = normalize_image(img_arr)
        return self.inference(img_arr_norm, other_arr)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'model_outputs': tf.TensorShape([self.num_outputs])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return f'{super().__str__()}-L:{self.seq_length}'


class Keras3D_CNN(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 seq_length=20,
                 num_outputs=2):
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        super().__init__(interpreter, input_shape)
        self.img_seq: List[np.ndarray] = []

    def seq_size(self) -> int:
        return self.seq_length

    def create_model(self):
        return build_3d_cnn(self.input_shape, s=self.seq_length,
                            num_outputs=self.num_outputs)

    def compile(self):
        self.interpreter.compile(loss='mse', optimizer=self.optimizer)

    def x_transform(self, records: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Return x from record, here x = stacked images """
        assert isinstance(records, list), 'List[TubRecord] expected'
        assert len(records) == self.seq_length, \
            f"Record list of length {self.seq_length} required but " \
            f"{len(records)} was passed"
        img_arrays = [rec.image(cached=True) for rec in records]
        return np.array(img_arrays)

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        """ Translates x into dictionary where all model input layer's names
            must be matched by dictionary keys. """
        img_arr = x
        return {'img_in': img_arr}

    def x_transform_and_process(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) -> XY:
        """ Transforms the record sequence into x for training the model to
            x, y. """
        img_seq = self.x_transform(record)
        assert isinstance(img_seq, np.ndarray), 'Expected np.ndarray'
        # apply augmentation / normalisation on sequence of images
        x_process = [img_processor(img) for img in img_seq]
        return np.array(x_process)

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) -> XY:
        """ Only return the last entry of angle/throttle"""
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, List[float]]]:
        assert isinstance(y, tuple), 'Expected tuple'
        return {'outputs': list(y)}

    def run(self, img_arr, other_arr=None):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)
        new_shape = (self.seq_length, *self.input_shape)
        img_arr = np.array(self.img_seq).reshape(new_shape)
        img_arr_norm = normalize_image(img_arr)
        return self.inference(img_arr_norm, other_arr)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'outputs': tf.TensorShape([self.num_outputs])})
        return shapes


class KerasLatent(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_latent(self.num_outputs, self.input_shape)

    def compile(self):
        loss = {"img_out": "mse", "n_outputs0": "mse", "n_outputs1": "mse"}
        weights = {"img_out": 100.0, "n_outputs0": 2.0, "n_outputs1": 1.0}
        self.interpreter.compile(optimizer=self.optimizer,
                                 loss=loss, loss_weights=weights)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[1]
        throttle = interpreter_out[2]
        return steering[0][0], throttle[0][0]


def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    """
    Helper function to create a standard valid-padded convolutional layer
    with square kernel and strides and unified naming convention

    :param filters:     channel dimension of the layer
    :param kernel:      creates (kernel, kernel) kernel matrix dimension
    :param strides:     creates (strides, strides) stride
    :param layer_num:   used in labelling the layer
    :param activation:  activation, defaults to relu
    :return:            tf.keras Convolution2D layer
    """
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))


def core_cnn_layers(img_in, drop, l4_stride=1):
    """
    Returns the core CNN layers that are shared among the different models,
    like linear, imu, behavioural

    :param img_in:          input layer of network
    :param drop:            dropout rate
    :param l4_stride:       4-th layer stride, default 1
    :return:                stack of CNN layers
    """
    x = img_in
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x


def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model


def default_memory(input_shape=(120, 160, 3), mem_length=3, mem_depth=0):
    drop = 0.2
    drop2 = 0.1
    logger.info(f'Creating memory model with length {mem_length}, depth '
                f'{mem_depth}')
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    mem_in = Input(shape=(2 * mem_length,), name='mem_in')
    y = mem_in
    for i in range(mem_depth):
        y = Dense(4 * mem_length, activation='relu', name=f'mem_{i}')(y)
        y = Dropout(drop2)(y)
    for i in range(1, mem_length):
        y = Dense(2 * (mem_length - i), activation='relu', name=f'mem_c_{i}')(y)
        y = Dropout(drop2)(y)
    x = concatenate([x, y])
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)
    activation = ['tanh', 'sigmoid']
    outputs = [Dense(1, activation=activation[i], name='n_outputs' + str(i))(x)
               for i in range(2)]
    model = Model(inputs=[img_in, mem_in], outputs=outputs, name='memory')
        outputs.append(Dense(1, activation='linear',
                             name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name="KerasLinear")
    return model


def linear_square_plus_cnn(x, size='S', is_seq=False):
    drop = 0.02
    # This makes the picture square in 1 steps (assuming 3x4 input) in all
    # following layers
    if size in ['XS', 'S', 'M']:
        filters = [16, 32, 64, 96]
        kernels = [(7, 7), (5, 5), (3, 3), (2, 2)]
        if size == 'M':
            filters += [128]
            kernels += [(2, 2)]
    elif size == 'L':  #
        filters = [20, 40, 80, 120, 160]
        kernels = [(9, 9), (7, 7), (5, 5), (3, 3), (2, 2)]
    else:  # size is R
        filters = [16, 32, 48, 64, 80]
        kernels = [(3, 3)] * 5
    if size in ['XS', 'S']:
        strides = [(3, 4), (2, 2)] + [(1, 1)] * 2
    elif size in ['M', 'L']:  # M or L
        strides = [(3, 4)] + [(1, 1)] * 4
    elif size == 'R':  # size is R
        strides = [(2, 2)] + [(1, 1)] * 4
    else:
        raise ValueError('Size must be in XS, X, M, L, R not ' + str(size))

    # build CNN layers with data as above and batch norm, pooling & dropout
    for i, f, k, s in zip(range(len(filters)), filters, kernels, strides):
        conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                      activation='relu', name='conv' + str(i))
        norm = BatchNormalization(name='batch_norm' + str(i))
        pool = MaxPooling2D(pool_size=(2, 2), padding='same',
                            name='pool' + str(i))
        dropout = Dropout(rate=drop, name='drop' + str(i))
        if is_seq:
            x = TD(conv, name='td_conv' + str(i))(x)
            x = TD(norm, name='td_norm' + str(i))(x)
            x = TD(pool, name='td_pool' + str(i))(x)
            x = TD(dropout, name='td_drop' + str(i))(x)
        else:
            x = conv(x)
            if size != 'R':
                x = norm(x)
                x = pool(x)
                x = dropout(x)
            else:
                if i < 3:
                    x = pool(x)
                elif i == 3:
                    x = MaxPooling2D(pool_size=(3, 3), padding='same',
                                     name='pool' + str(i))(x)

    flat = Flatten(name='flattened')
    x = TD(flat, name='td_flat')(x) if is_seq else flat(x)
    return x


def square_plus_dense(size='S'):
    d = dict(XS=[72] * 3 + [48],
             S=[96] * 4 + [48],
             M=[128] * 5 + [64],
             L=[144] * 8,
             R=[128] * 5 + [64])
    if size not in d:
        raise ValueError('size must be in', d.keys(), 'but was', size)
    return d[size]


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
    inputs = [img_in]
    model = square_plus_output_layers(x, size, l2, inputs, imu_dim=None,
                                      seq_len=seq_len)
    return model


def linear_square_plus_imu(input_shape=(120, 160, 3), roi_crop=(0, 0),
                           imu_dim=6, size='S', seq_len=None):
    assert 0 < imu_dim <= 6, 'imu_dim must be number in [1,..,6]'
    l2 = 0.001
    input_shape = adjust_input_shape(input_shape, roi_crop)
    imu_shape = (imu_dim,)
    if seq_len:
        input_shape = (seq_len,) + input_shape
        imu_shape = (seq_len,) + imu_shape
    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=imu_shape, name="imu_in")
    x = img_in
    x = linear_square_plus_cnn(x, size, seq_len is not None)
    y = imu_in
    imu_dense_size = dict(XS=20, S=24, M=36, L=48, R=36)
    imu_dense = Dense(units=imu_dense_size[size], activation='relu',
                      kernel_regularizer=regularizers.l2(l2),
                      name='dense_imu')
    y = TD(imu_dense, name='td_imu_dense')(y) if seq_len else imu_dense(y)
    z = concatenate([x, y])
    inputs = [img_in, imu_in]
    model = square_plus_output_layers(z, size, l2, inputs, imu_dim, seq_len)
    return model


def default_categorical(input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop, l4_stride=2)
    x = Dense(100, activation='relu', name="dense_1")(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name="dense_2")(x)
    x = Dropout(drop)(x)
    # Categorical output of the angle into 15 bins
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    # categorical output of throttle into 20 bins
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out],
                  name='categorical')
    return model

def square_plus_output_layers(in_tensor, size, l2, model_inputs,
                              imu_dim=None, seq_len=None):
    layers = square_plus_dense(size)
    z = in_tensor
    for i, l in zip(range(len(layers)), layers):
        if seq_len:
            z = LSTM(units=l,
                     kernel_regularizer=regularizers.l2(l2),
                     recurrent_regularizer=regularizers.l2(l2),
                     name='lstm' + str(i),
                     return_sequences=(i != len(layers) - 1))(z)
        else:
            z = Dense(units=l, activation='relu',
                      kernel_regularizer=regularizers.l2(l2),
                      name='dense' + str(i))(z)

    angle_out = Dense(units=1, activation='linear', name='angle')(z)
    throttle_out = Dense(units=1, activation='linear', name='throttle')(z)
    if len(model_inputs) > 1:
        name = 'SquarePlusImu_' + size + '_' + str(imu_dim)
    else:
        name = 'SquarePlus_' + size
    if seq_len:
        name += '_lstm_' + str(seq_len)
    model = Model(inputs=model_inputs,
                  outputs=[angle_out, throttle_out],
                  name=name)
    return model


def default_imu(num_outputs, num_imu_inputs, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = core_cnn_layers(img_in, drop)
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
        
    model = Model(inputs=[img_in, imu_in], outputs=outputs, name='imu')
    return model


def default_bhv(num_bvh_inputs, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    # tensorflow is ordering the model inputs alphabetically in tensorrt,
    # so behavior must come after image, hence we put an x here in front.
    bvh_in = Input(shape=(num_bvh_inputs,), name="xbehavior_in")

    x = core_cnn_layers(img_in, drop)
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
    
    # Categorical output of the angle into 15 bins
    angle_out = Dense(15, activation='softmax', name='angle_out')(z)
    # Categorical output of throttle into 20 bins
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(z)

    model = Model(inputs=[img_in, bvh_in], outputs=[angle_out, throttle_out],
                  name='behavioral')
    return model


def default_loc(num_locations, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    
    z = Dense(50, activation='relu')(x)
    z = Dropout(drop)(z)

    # linear output of the angle
    angle_out = Dense(1, activation='linear', name='angle')(z)
    # linear output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle')(z)
    # Categorical output of location
    # Here is a crazy detail b/c TF Lite has a bug and returns the outputs
    # in the alphabetical order of the name of the layers, so make sure
    # this output comes last
    loc_out = Dense(num_locations, activation='softmax', name='zloc')(z)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out, loc_out],
                  name='localizer')
    return model


def rnn_lstm(seq_length=3, num_outputs=2, input_shape=(120, 160, 3)):
    # add sequence length dimensions as keras time-distributed expects shape
    # of (num_samples, seq_length, input_shape)
    img_seq_shape = (seq_length,) + input_shape
    img_in = Input(shape=img_seq_shape, name='img_in')
    drop_out = 0.3

    x = img_in
    x = TD(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TD(Flatten(name='flattened'))(x)
    x = TD(Dense(100, activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)

    x = LSTM(128, return_sequences=True, name="LSTM_seq")(x)
    x = Dropout(.1)(x)
    x = LSTM(128, return_sequences=False, name="LSTM_fin")(x)
    x = Dropout(.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    out = Dense(num_outputs, activation='linear', name='model_outputs')(x)
    model = Model(inputs=[img_in], outputs=[out], name='lstm')
    return model


def build_3d_cnn(input_shape, s, num_outputs):
    """
    Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model/model.py

    :param input_shape:     image input shape
    :param s:               sequence length
    :param num_outputs:     output dimension
    :return:                keras model
    """
    drop = 0.5
    input_shape = (s, ) + input_shape
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    # Second layer
    x = Conv3D(
            filters=16, kernel_size=(3, 3, 3), strides=(1, 3, 3),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Third layer
    x = Conv3D(
            filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
        data_format=None)(x)
    # Fourth layer
    x = Conv3D(
            filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Fifth layer
    x = Conv3D(
            filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Fully connected layer
    x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    out = Dense(num_outputs, name='outputs')(x)
    model = Model(inputs=[img_in], outputs=out, name='3dcnn')
    return model


def default_latent(num_outputs, input_shape):
    # TODO: this auto-encoder should run the standard cnn in encoding and
    #  have corresponding decoder. Also outputs should be reversed with
    #  images at end.
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, 5, strides=2, activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_6")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_7")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 1, strides=2, activation='relu', name="latent")(x)

    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_1")(x)
    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_2")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_3")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_4")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_5")(y)
    y = Conv2DTranspose(filters=1, kernel_size=3, strides=2, name="img_out")(y)
    
    x = Flatten(name='flattened')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = [y]
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs, name='latent')
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

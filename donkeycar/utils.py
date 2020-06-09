'''
utils.py

Functions that don't fit anywhere else.

'''
import glob
import itertools
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
import zipfile
import json
import pandas as pd
from io import BytesIO

import numpy as np
from PIL import Image

'''
IMAGES
'''
one_byte_scale = 1.0 / 255.0


def scale(im, size=128):
    '''
    accepts: PIL image, size of square sides
    returns: PIL image scaled so sides lenght = size 
    '''
    size = (size,size)
    im.thumbnail(size, Image.ANTIALIAS)
    return im


def img_to_binary(img, format='jpeg'):
    '''
    accepts: PIL image
    returns: binary stream (used to save to database)
    '''
    f = BytesIO()
    try:
        img.save(f, format=format)
    except Exception as e:
        raise e
    return f.getvalue()


def arr_to_binary(arr):
    '''
    accepts: numpy array with shape (Hight, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    img = arr_to_img(arr)
    return img_to_binary(img)


def arr_to_img(arr):
    '''
    accepts: numpy array with shape (Height, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    arr = np.uint8(arr)
    img = Image.fromarray(arr)
    return img


def img_to_arr(img):
    '''
    accepts: numpy array with shape (Height, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    return np.array(img)


def binary_to_img(binary):
    '''
    accepts: binary file object from BytesIO
    returns: PIL image
    '''
    if binary is None or len(binary) == 0:
        return None

    img = BytesIO(binary)
    try:
        img = Image.open(img)
        return img
    except:
        return None


def norm_img(img):
    return (img - img.mean() / np.std(img)) * one_byte_scale


def create_video(img_dir_path, output_video_path):
    import envoy
    # Setup path to the images with telemetry.
    full_path = os.path.join(img_dir_path, 'frame_*.png')

    # Run ffmpeg.
    command = ("""ffmpeg
               -framerate 30/1
               -pattern_type glob -i '%s'
               -c:v libx264
               -r 15
               -pix_fmt yuv420p
               -y
               %s""" % (full_path, output_video_path))
    response = envoy.run(command)


def rgb2gray(rgb):
    '''
    take a numpy rgb image return a new single channel image converted to greyscale
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def img_crop(img_arr, top, bottom):
    
    if bottom == 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]


def normalize_and_crop(img_arr, cfg):
    img_arr = img_arr.astype(np.float32) * one_byte_scale
    if cfg.ROI_CROP_TOP or cfg.ROI_CROP_BOTTOM:
        img_arr = img_crop(img_arr, cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM)
        if len(img_arr.shape) == 2:
            img_arrH = img_arr.shape[0]
            img_arrW = img_arr.shape[1]
            img_arr = img_arr.reshape(img_arrH, img_arrW, 1)
    return img_arr


def load_scaled_image_arr(filename, cfg):
    '''
    load an image from the filename, and use the cfg to resize if needed
    also apply cropping and normalize
    '''
    import donkeycar as dk
    try:
        img = Image.open(filename)
        if img.height != cfg.IMAGE_H or img.width != cfg.IMAGE_W:
            img = img.resize((cfg.IMAGE_W, cfg.IMAGE_H))
        img_arr = np.array(img)
        img_arr = normalize_and_crop(img_arr, cfg)
        croppedImgH = img_arr.shape[0]
        croppedImgW = img_arr.shape[1]
        if img_arr.shape[2] == 3 and cfg.IMAGE_DEPTH == 1:
            img_arr = dk.utils.rgb2gray(img_arr).reshape(croppedImgH, croppedImgW, 1)
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr


'''
FILES
'''


def most_recent_file(dir_path, ext=''):
    '''
    return the most recent file given a directory path and extension
    '''
    query = dir_path + '/*' + ext
    newest = min(glob.iglob(query), key=os.path.getctime)
    return newest


def make_dir(path):
    real_path = os.path.expanduser(path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    return real_path


def zip_dir(dir_path, zip_path):
    """ 
    Create and save a zipfile of a one level directory
    """
    file_paths = glob.glob(dir_path + "/*") #create path to search for files.
    
    zf = zipfile.ZipFile(zip_path, 'w')
    dir_name = os.path.basename(dir_path)
    for p in file_paths:
        file_name = os.path.basename(p)
        zf.write(p, arcname=os.path.join(dir_name, file_name))
    zf.close()
    return zip_path



'''
BINNING
functions to help converte between floating point numbers and categories.
'''


def clamp(n, min, max):
    if n < min:
        return min
    if n > max:
        return max
    return n


def linear_bin(a, N=15, offset=1, R=2.0):
    '''
    create a bin of length N
    map val A to range R
    offset one hot bin by offset, commonly R/2
    '''
    a = a + offset
    b = round(a / (R/(N-offset)))
    arr = np.zeros(N)
    b = clamp(b, 0, N - 1)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr, N=15, offset=-1, R=2.0):
    '''
    preform inverse linear_bin, taking
    one hot encoded arr, and get max value
    rescale given R range and offset
    '''
    b = np.argmax(arr)
    a = b *(R/(N + offset)) + offset
    return a


def map_range(x, X_min, X_max, Y_min, Y_max):
    ''' 
    Linear mapping between two ranges of values 
    '''
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    YX_ratio = Y_range / X_range

    y = ((x-X_min) * YX_ratio + Y_min)

    return int(y)

'''
ANGLES
'''


def norm_deg(theta):
    while theta > 360:
        theta -= 360
    while theta < 0:
        theta += 360
    return theta


DEG_TO_RAD = math.pi / 180.0


def deg2rad(theta):
    return theta * DEG_TO_RAD

'''
VECTORS
'''


def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))


'''
NETWORKING
'''

def my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('192.0.0.8', 1027))
    return s.getsockname()[0]


'''
OTHER
'''

def map_frange(x, X_min, X_max, Y_min, Y_max):
    '''
    Linear mapping between two ranges of values
    '''
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range/Y_range

    y = ((x-X_min) / XY_ratio + Y_min)

    return y


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def param_gen(params):
    '''
    Accepts a dictionary of parameter options and returns 
    a list of dictionary with the permutations of the parameters.
    '''
    for p in itertools.product(*params.values()):
        yield dict(zip(params.keys(), p ))


def run_shell_command(cmd, cwd=None, timeout=15):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    out = []
    err = []

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill(proc.pid)

    for line in proc.stdout.readlines():
        out.append(line.decode())

    for line in proc.stderr.readlines():
        err.append(line)
    return out, err, proc.pid


def kill(proc_id):
    os.kill(proc_id, signal.SIGINT)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


"""
Tub management
"""


def expand_path_masks(paths, exclude=None):
    '''
    take a list of paths and expand any wildcards
    returns a new list of paths fully expanded
    '''
    import glob
    from braceexpand import braceexpand
    expanded_paths = []
    for path in paths:
        if '*' in path or '?' in path or '[' in path or '{' in path:
            # brace expand first
            if '{' in path:
                path = list(braceexpand(path))
            else:
                path = [path]
            for ex_path in path:
                mask_paths = glob.glob(ex_path)
                expanded_paths += mask_paths
        else:
            expanded_paths.append(path)
    # finally allow exclusion patterns on expanded paths
    if exclude is None:
        exclude = []
    elif type(exclude) is str:
        exclude = [exclude]
    filtered_paths = []
    for path in expanded_paths:
        if not any([ex in path for ex in exclude]):
            filtered_paths.append(path)
    return filtered_paths


def gather_tub_paths(cfg, tub_names=None, exclude=None):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub paths
    '''
    if tub_names:
        if type(tub_names) == list:
            tub_paths = [os.path.expanduser(n) for n in tub_names]
        else:
            tub_paths = [os.path.expanduser(n) for n in tub_names.split(',')]
        return expand_path_masks(tub_paths, exclude=exclude)
    else:
        paths = [os.path.join(cfg.DATA_PATH, n) for n in os.listdir(cfg.DATA_PATH)]
        dir_paths = []
        for p in paths:
            if os.path.isdir(p):
                dir_paths.append(p)
        return dir_paths


def gather_tubs(cfg, tub_names, exclude=None):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub objects initialized to each path
    '''
    from donkeycar.parts.datastore import Tub
    
    tub_paths = gather_tub_paths(cfg, tub_names, exclude)
    tubs = [Tub(p) for p in tub_paths]

    return tubs

"""
Training helpers
"""


def get_image_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[0])


def get_record_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[1].split('.')[0])


def gather_records(cfg, tub_names, exclude=None, verbose=False, data_base=None):
    tubs = gather_tubs(cfg, tub_names, exclude)
    records = []
    tub_paths = []
    exclude_laps = getattr(cfg, 'EXCLUDE_SLOW_LAPS', None)
    exclude_by = getattr(cfg, 'SORT_LAPS_BY', 'lap_time')

    for tub in tubs:
        if verbose:
            print(tub.path)
        if exclude_laps is not None:
            tub.exclude_slow_laps(cfg.EXCLUDE_SLOW_LAPS, sort_by=exclude_by)
        record_paths = tub.gather_records()
        records += record_paths
        tub_paths.append(tub.path)

    if data_base is not None:
        data_base['ExcludeLaps'] = exclude_laps
        data_base['ExcludeBy'] = exclude_by
        data_base['Tubs'] = ",".join(tub_paths)

    return records


def get_model_by_type(model_type, cfg):
    '''
    given the string model_type and the configuration settings in cfg
    create a Keras model and return it.
    '''
    from donkeycar.parts.keras import KerasRNN_LSTM, KerasBehavioral, \
        KerasCategorical, KerasIMU, KerasLinear, KerasSquarePlus, \
        KerasSquarePlusImu, Keras3D_CNN, KerasLocalizer, KerasLatent
    from donkeycar.parts.tflite import TFLitePilot
 
    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    print("\"get_model_by_type\" model Type is: {}".format(model_type))

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    roi_crop = (cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM)

    if model_type == "tflite_linear":
        kl = TFLitePilot()
    elif model_type == "localizer" or cfg.TRAIN_LOCALIZER:
        kl = KerasLocalizer(num_locations=cfg.NUM_LOCATIONS, input_shape=input_shape)
    elif model_type == "behavior" or cfg.TRAIN_BEHAVIORS:
        kl = KerasBehavioral(num_outputs=2, num_behavior_inputs=len(cfg.BEHAVIOR_LIST), input_shape=input_shape)        
    elif model_type == "imu":
        kl = KerasIMU(num_outputs=2, num_imu_inputs=6, input_shape=input_shape)        
    elif model_type == "linear":
        kl = KerasLinear(input_shape=input_shape, roi_crop=roi_crop)
    elif model_type == "square_plus":
        nn_size = cfg.NN_SIZE if hasattr(cfg, 'NN_SIZE') else 'S'
        kl = KerasSquarePlus(input_shape=input_shape, roi_crop=roi_crop,
                             size=nn_size)
    elif model_type == "square_plus_imu":
        imu_dim = cfg.IMU_DIM if hasattr(cfg, 'IMU_DIM') else 6
        nn_size = cfg.NN_SIZE if hasattr(cfg, 'NN_SIZE') else 'S'
        kl = KerasSquarePlusImu(input_shape=input_shape, roi_crop=roi_crop,
                                imu_dim=imu_dim, size=nn_size)
    elif model_type == "tensorrt_linear":
        # Aggressively lazy load this. This module imports pycuda.autoinit which causes a lot of unexpected things
        # to happen when using TF-GPU for training.
        from donkeycar.parts.tensorrt import TensorRTLinear
        kl = TensorRTLinear(cfg=cfg)
    elif model_type == "coral_tflite_linear":
        from donkeycar.parts.coral import CoralLinearPilot
        kl = CoralLinearPilot()
    elif model_type == "3d":
        kl = Keras3D_CNN(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, seq_length=cfg.SEQUENCE_LENGTH)
    elif model_type == "rnn":
        kl = KerasRNN_LSTM(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, seq_length=cfg.SEQUENCE_LENGTH)
    elif model_type == "categorical":
        kl = KerasCategorical(input_shape=input_shape, throttle_range=cfg.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE, roi_crop=roi_crop)
    elif model_type == "latent":
        kl = KerasLatent(input_shape=input_shape)
    elif model_type == "fastai":
        from donkeycar.parts.fastai import FastAiPilot
        kl = FastAiPilot()
    else:
        model_types = ['tflite_linear', 'localizer', 'behavior', 'imu', 'linear', 'square_plus', 'square_plus_imu',
                       'tensorrt_linear', 'coral_tflite_linear', '3d', 'rnn', 'categorical', 'latent', 'fastai']
        raise ValueError("Unknown model type: '{:}', known types: {:}. Note for TFlite models pass 'tflite_linear' "
                         "whatever the underlying model is.".format(model_type, model_types))

    return kl


def get_test_img(model):
    '''
    query the input to see what it likes
    make an image capable of using with that test model
    '''
    assert(len(model.inputs) > 0)
    try:
        count, h, w, ch = model.inputs[0].get_shape()
        seq_len = 0
    except Exception as e:
        count, seq_len, h, w, ch = model.inputs[0].get_shape()

    # generate random array in the right shape
    img = np.random.rand(int(h), int(w), int(ch))

    return img


def train_test_split(data_list, shuffle=True, test_size=0.2):
    '''
    take a list, split it into two sets while selecting a 
    random element in order to shuffle the results.
    use the test_size to choose the split percent.
    shuffle is always True, left there to be backwards compatible
    '''
    assert shuffle
    train_data = []
    target_train_size = len(data_list) * (1. - test_size)
    i_sample = 0

    while i_sample < target_train_size and len(data_list) > 1:
        i_choice = random.randint(0, len(data_list) - 1)
        train_data.append(data_list.pop(i_choice))
        i_sample += 1

    # remainder of the original list is the validation set
    val_data = data_list

    return train_data, val_data


"""
Timers
"""


class FPSTimer(object):
    def __init__(self):
        self.t = time.time()
        self.iter = 0

    def reset(self):
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            e = time.time()
            print('fps', 100.0 / (e - self.t))
            self.t = time.time()
            self.iter = 0

"""
Pilot management
"""


def make_pilot_databases(model_path):
    """
    Return two dataframes containing pilot traingin info
    :param model_path:
    :return: pandas dataframe tuple
    """
    files = os.listdir(model_path)
    j_data = []
    for d in files:
        if '.json' in d and d[0] is not '.':
            full_name = os.path.join(model_path, d)
            with open(full_name, 'r') as f:
                j_data.append(json.load(f))
    if not j_data:
        print('No .json pilots found')
        return pd.DataFrame(), pd.DataFrame()
    df_pilots = pd.DataFrame(j_data)
    df_pilots = df_pilots.set_index('Num')
    tubs = df_pilots['Tubs'].drop_duplicates()
    multi_tubs = [tub for tub in tubs if ',' in tub]
    d = dict(zip(multi_tubs,
                 ['tub_group_' + str(i) for i in range(len(multi_tubs))]))
    new_tubs = [d[tub] if tub in d else tub for tub in df_pilots['Tubs']]
    df_pilots['Tubs'] = new_tubs
    df_pilots.sort_index(inplace=True)
    # pandas explode normalises multiplicity of arrays as entries in data frame
    df_tubs = pd.DataFrame(zip(d.values(), [k.split(',') for k in d.keys()]),
                           columns=['TubGroup', 'Tubs']).explode('Tubs')
    return df_pilots, df_tubs

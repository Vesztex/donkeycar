#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:32:53 2017

@author: wroscoe
"""
import os
import sys
import time
import json
import datetime
import random
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil

from donkeycar.parts.augment import augment_pil_image
from donkeycar.utils import arr_to_img
from donkeycar.parts.transform import ImgBrightnessNormaliser


class Tub(object):
    """
    A data store to store sensor data in a key, value format. Accepts str, int,
    float, image_array, image, and array data types. For example:

    #Create a tub to store speed values.
    path = '~/mydonkey/test_tub'
    inputs = ['user/speed', 'cam/image']
    types = ['float', 'image']
    t=Tub(path=path, inputs=inputs, types=types)
    """

    def __init__(self, path, inputs=None, types=None,
                 user_meta=[], allow_reverse=True):

        self.path = os.path.expanduser(path)
        if self.path[-1] == '/':
            self.path = self.path[:-1]
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.exclude_path = os.path.join(self.path, "exclude.json")
        self.df = None
        exists = os.path.exists(self.path)
        self.allow_reverse = allow_reverse

        if exists:
            # load log and meta
            print("Tub exists: {}".format(self.path))
            try:
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
            except FileNotFoundError:
                self.meta = {'inputs': [], 'types': []}

            try:
                with open(self.exclude_path, 'r') as f:
                    excl = json.load(f)  # stored as a list
                    self.exclude = set(excl)
            except FileNotFoundError:
                self.exclude = set()

            try:
                self.current_ix = self.get_last_ix() + 1
            except ValueError:
                self.current_ix = 0

            if 'start' in self.meta:
                self.start_time = self.meta['start']
            else:
                self.start_time = time.time()
                self.meta['start'] = self.start_time

        elif not exists and inputs:
            print('Tub does NOT exist. Creating new tub...')
            self.start_time = time.time()
            # create log and save meta
            os.makedirs(self.path)
            self.meta = {'inputs': inputs,
                         'types': types,
                         'start': self.start_time}
            for kv in user_meta:
                kvs = kv.split(":")
                if len(kvs) == 2:
                    self.meta[kvs[0]] = kvs[1]
                # else exception? print message?
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            self.exclude = set()
            print('New tub created at: {}'.format(self.path))

        else:
            msg = "The tub path " + path + " you provided doesn't exist and " \
                "you didnt pass any meta info (inputs & types) to create a " \
                "new tub. Please check your tub path or provide meta info to " \
                "create a new tub."

            raise AttributeError(msg)

    def get_last_ix(self):
        index = self.get_index()
        return max(index)

    def update_df(self):
        index = self.get_index(shuffled=False)
        df = pd.DataFrame([self.get_json_record(i, unravel=True) for i in
                           index])
        df = df.set_index(pd.Index(index))
        self.df = df

    def get_df(self):
        if self.df is None:
            self.update_df()
        return self.df

    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6] == 'record']

        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]

        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)

        return nums

    @property
    def inputs(self):
        return list(self.meta['inputs'])

    @property
    def types(self):
        return list(self.meta['types'])

    def get_input_type(self, key):
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def write_json_record(self, json_data):
        path = self.get_json_record_path(self.current_ix)
        try:
            with open(path, 'w') as fp:
                json.dump(json_data, fp)

        except TypeError:
            print('troubles with record:', json_data)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_num_records(self):
        import glob
        files = glob.glob(os.path.join(self.path, 'record_*.json'))
        return len(files)

    def make_record_paths_absolute(self, record_dict):
        # make paths absolute
        d = {}
        for k, v in record_dict.items():
            if self.get_input_type(k) == 'image_array':  # path to jpg file
                v = os.path.join(self.path, v)
            d[k] = v
        return d

    def copy(self, suffix='copy'):
        """
        Create new tub by inserting 'copy' before the date like
        'tub_XY_YY-MM-DD' -> 'tub_XY_copy_YY-MM-DD
        :param suffix:  string to insert into copied tub name
        :return:        new tub at new location
        """
        tub_path = self.path
        # remove trailing slash if exits
        if tub_path[-1] == '/':
            tub_path = tub_path[:-1]
        head, tail = os.path.split(tub_path)
        tail_list = tail.split('_')
        length = len(tail_list)
        tail_list.insert(length - 1, suffix)
        new_tail = '_'.join(tail_list)
        new_path = os.path.join(head, new_tail)
        # copy whole tub to new location
        shutil.copytree(self.path, new_path)
        new_tub = Tub(new_path)
        return new_tub

    def check(self, fix=False):
        """
        Iterate over all records and make sure we can load them.
        Optionally remove records that cause a problem.
        """
        print('Checking tub:%s.' % self.path)
        print('Found: %d records.' % self.get_num_records())
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except Exception as e:
                problems = True
                if fix is False:
                    print('Problem at {}: {}'.format(self.path, str(e)))
                else:
                    print('Removing record at {}, because: {}'
                          .format(self.path, str(e)))
                    self.remove_record(ix)
        if not problems:
            print("No problems found.")

    def remove_record(self, ix):
        """
        remove data associate with a record, both json and jpg
        """
        try:
            record = self.get_json_record_path(ix)
            os.remove(record)
        except FileNotFoundError:
            pass
        jpeg = self.make_file_name('cam-image_array', '.jpg', ix=ix)
        full_path_jpeg = os.path.join(self.path, jpeg)
        try:
            os.remove(full_path_jpeg)
        except OSError:
            pass

    def remove_last_records(self, num_records):
        """
        Removes records from the end. Assume these are continuously labeled.
        :param num_records: number or records to be removed from end
        """
        removed_records = 0
        last_ix = self.current_ix
        while removed_records < num_records:
            self.remove_record(last_ix - removed_records)
            removed_records += 1
        self.current_ix = last_ix - removed_records
        print('Removed records {} - {} from tub'
              .format(self.current_ix + 1, last_ix))

    def put_record(self, data):
        """
        Save values like images that can't be saved in the csv log and
        return a record with references to the saved values that can
        be saved in a csv.
        """
        json_data = {}
        self.current_ix += 1

        for key, val in data.items():
            typ = self.get_input_type(key)

            if (val is not None) and (typ == 'float'):
                # in case val is a numpy.float32, which json doesn't like
                json_data[key] = float(val)

            elif typ in ['str', 'float', 'int', 'boolean', 'vector']:
                json_data[key] = val

            elif typ == 'image':
                path = self.make_file_path(key)
                val.save(path)
                json_data[key] = path

            elif typ == 'image_array':
                img = Image.fromarray(np.uint8(val))
                name = self.make_file_name(key, ext='.jpg')
                img.save(os.path.join(self.path, name))
                json_data[key] = name

            elif typ == 'gray16_array':
                # save np.uint16 as a 16bit png
                img = Image.fromarray(np.uint16(val))
                name = self.make_file_name(key, ext='.png')
                img.save(os.path.join(self.path, name))
                json_data[key]=name

            else:
                msg = 'Tub does not know what to do with key {} of type {}. ' \
                      'Current ix {}, input data: {}'\
                      .format(key, typ, self.current_ix, data)
                raise TypeError(msg)

        json_data['milliseconds'] = int((time.time() - self.start_time) * 1000)
        self.write_json_record(json_data)
        return self.current_ix

    def erase_last_n_records(self, num_erase):
        """
        erase N records from the disc and move current back accordingly
        """
        last_erase = self.current_ix
        first_erase = last_erase - num_erase
        self.current_ix = first_erase - 1
        if self.current_ix < 0:
            self.current_ix = 0

        for i in range(first_erase, last_erase):
            if i < 0:
                continue
            self.erase_record(i)

    def erase_record(self, i):
        json_path = self.get_json_record_path(i)
        if os.path.exists(json_path):
            os.unlink(json_path)
        img_filename = '%d_cam-image_array_.jpg' % i
        img_path = os.path.join(self.path, img_filename)
        if os.path.exists(img_path):
            os.unlink(img_path)

    def get_json_record_path(self, ix):
        return os.path.join(self.path, 'record_' + str(ix) + '.json')

    def get_json_record(self, ix, unravel=False):
        path = self.get_json_record_path(ix)
        err_add = 'You may want to run `donkey tubcheck --fix`'
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception(('Bad record: %d. ' + err_add) % ix)
        except FileNotFoundError:
            raise

        # if negative throttle values are recorded
        if self.allow_reverse is False and json_data["user/throttle"] < 0.0:
            raise Exception(('Bad record: %d. "user/throttle" should be >0 for '
                            'recorded data. ' + err_add) % ix)

        # if negative or zero car speed values are recorded
        if "car/speed" in json_data and json_data["car/speed"] <= 0.0:
            raise Exception(('Bad record: %d. "car/speed" should be >0 for '
                            'recorded data. ' + err_add) % ix)

        if unravel:
            unravel_dict = {}
            delete_keys = []
            for k, v in json_data.items():
                typ = self.get_input_type(k)
                if typ == 'vector':
                    for i in range(len(v)):
                        unravel_dict[k + "_" + str(i)] = v[i]
                    delete_keys.append(k)
            for d in delete_keys:
                del(json_data[d])
            json_data.update(unravel_dict)

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict

    def get_record(self, ix):
        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data

    def read_record(self, record_dict):
        data = {}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)
            # load objects that were saved as separate files
            if typ == 'image_array':
                img = Image.open(val)
                val = np.array(img)
            data[key] = val
        return data

    def gather_records(self):
        ri = lambda fnm: int(os.path.basename(fnm).split('_')[1].split('.')[0])
        record_paths = glob.glob(os.path.join(self.path, 'record_*.json'))
        if len(self.exclude) > 0:
            record_paths = [f for f in record_paths if ri(f) not in self.exclude]
        record_paths.sort(key=ri)
        return record_paths

    def make_file_name(self, key, ext='.png', ix=None):
        this_ix = ix
        if this_ix is None:
            this_ix = self.current_ix
        name = '_'.join([str(this_ix), key, ext])
        name = name.replace('/', '-')
        return name

    def delete(self):
        """ Delete the folder and files for this tub. """
        import shutil
        shutil.rmtree(self.path)

    def excluded(self, index):
        return index in self.exclude

    def exclude_index(self, index):
        self.exclude.add(index)

    def include_index(self, index):
        try:
            self.exclude.remove(index)
        except:
            pass

    def make_lap_times(self):
        """
        Method returns a dataframe with lap numbers and times
        :return: dataframe
        """
        df = self.get_df()
        assert 'car/lap' in df.columns, 'No lap data found in tub ' + self.path
        laps = df['car/lap'].unique()
        times = []
        distances = []
        gyro_z_acc = []
        accel_x_acc = []
        last_start = None
        last_start_dist = None
        has_distance = 'car/distance' in df.columns
        for l in laps[1:]:
            mask = df['car/lap'] == l
            lap_df = df[mask]
            start = lap_df['milliseconds'].iloc[0]
            if has_distance:
                start_dist = lap_df['car/distance'].iloc[0]
            if last_start is not None:
                times.append((start - last_start) * 1.0e-3)
                if has_distance:
                    lap_dist = start_dist - last_start_dist
                    distances.append(lap_dist)
                gyro_z = lap_df['car/gyro_2']
                accel_x = lap_df['car/accel_0']
                # use average absolute turning speed to indicate fuzzy
                # driving or alternatively absolute x (i.e. left/right)
                # acceleration
                gyro_z_acc.append(gyro_z.abs().mean())
                accel_x_acc.append(accel_x.abs().mean())
            last_start = start
            if has_distance:
                last_start_dist = start_dist

        data = dict(lap=laps[1:-1],
                    lap_time=times,
                    accel_x=accel_x_acc,
                    gyro_z=gyro_z_acc)
        if has_distance:
            data['lap_distances'] = distances
        return pd.DataFrame(data, index=laps[1: -1])

    def exclude_slow_laps(self, keep_frac_or_seconds=None, clean=True,
                          sort_by='lap_time'):
        """
        Removes records of slower laps
        :param sort_by:                 sort by which column
        :param clean:                   remove laps that are < 0.9 median length
        :param keep_frac_or_seconds:    either fraction of the laps to keep or
                                        laps with are faster than the  argument
                                        as seconds. Interpreted as fraction if
                                        <= 1  and as seconds if > 1
        :return:                        laps to keep array
        """
        # if None do nothing
        if keep_frac_or_seconds is None:
            return None
        # if exclude set is non-empty, empty it first
        if self.exclude:
            self.exclude.clear()
        df = self.make_lap_times()
        all_laps = len(df)
        assert sort_by in df, "Cannot find column " + sort_by + " for sorting"
        print('Tub {:}: '.format(self.path), end='')
        # remove laps that are too short
        if clean and 'lap_distances' in df:
            med_dist = df['lap_distances'].median()
            df = df[(df['lap_distances'] > 0.9 * med_dist) &
                    (df['lap_distances'] < 1.1 * med_dist)]
            print('retained {} from {} laps. '.format(len(df), all_laps),
                  end='')

        if keep_frac_or_seconds <= 1:
            df_sorted = df.sort_values(by=[sort_by])
            pct = (1.0 - keep_frac_or_seconds) * 100.0
            text = 'more than {:.1f}% quantile of {:}'.format(pct, sort_by)
            slowest_index = int(len(df) * keep_frac_or_seconds)
            laps_to_keep = df_sorted['lap'].iloc[:slowest_index]

        else:
            laps_to_keep = df.loc[df[sort_by] <= keep_frac_or_seconds]['lap']
            text = '{:} with values > {:.2f}s'.format(sort_by,
                                                      keep_frac_or_seconds)

        df_records = self.get_df()
        df_to_remove = df_records[~df_records['car/lap'].isin(laps_to_keep)]
        self.exclude = set(df_to_remove.index)
        print('Exclude {:.2f}% records which are {}'
              .format(len(self.exclude) / len(df_records) * 100.0, text))
        return laps_to_keep

    def augment_images(self):
        """
        Augment all images inplace
        """
        # define the processor
        def processor(img_arr):
            # here val is an img_arr
            img = arr_to_img(img_arr)
            # then augment and denormalise
            img_aug = augment_pil_image(img)
            return img_aug
        self._process_images(processor)

    def normalize_brightness_in_images(self, norm):
        """
        Normalises image brightness of all images in place
        :param norm: normalisation factor [0, 255]
        """
        # define the brightness normaliser
        img_br = ImgBrightnessNormaliser(norm)

        # define the processor
        def processor(img_arr):
            out_arr = img_br.run(img_arr)
            return arr_to_img(out_arr)

        self._process_images(processor)

    def _process_images(self, processor):
        """
        Go through all images and process them
        :param processor: function(img_arr) returning an PIL image
        """
        # Get all record's index
        index = self.get_index(shuffled=False)
        print('Processing', len(index), 'images in', self.path)
        # Go through index
        for ix in tqdm(index):
            data = self.get_record(ix)
            for key, val in data.items():
                typ = self.get_input_type(key)
                # load objects that were saved as separate files
                if typ == 'image_array':
                    # here val is an img_arr
                    img_out = processor(val)
                    name = self.make_file_name(key, ext='.jpg', ix=ix)
                    try:
                        img_out.save(os.path.join(self.path, name))
                    except IOError as err:
                        print(err)

    def write_exclude(self):
        if 0 == len(self.exclude):
            # If the exclude set is empty don't leave an empty file around.
            if os.path.exists(self.exclude_path):
                os.unlink(self.exclude_path)
        else:
            with open(self.exclude_path, 'w') as f:
                json.dump(list(self.exclude), f)

    def get_record_gen(self, record_transform=None, shuffle=True, df=None):

        if df is None:
            df = self.get_df()

        while True:
            for _, row in self.df.iterrows():
                if shuffle:
                    record_dict = df.sample(n=1).to_dict(orient='record')[0]
                else:
                    record_dict = row

                if record_transform:
                    record_dict = record_transform(record_dict)

                record_dict = self.read_record(record_dict)
                yield record_dict

    def get_batch_gen(self, keys, record_transform=None, batch_size=128,
                      shuffle=True, df=None):

        record_gen = self.get_record_gen(record_transform, shuffle=shuffle,
                                         df=df)

        if keys is None:
            keys = list(self.df.columns)

        while True:
            record_list = []
            for _ in range(batch_size):
                record_list.append(next(record_gen))
            batch_arrays = {}
            for i, k in enumerate(keys):
                arr = np.array([r[k] for r in record_list])
                batch_arrays[k] = arr
            yield batch_arrays

    def get_train_gen(self, x_keys, y_keys, batch_size=128,
                      record_transform=None, df=None):
        batch_gen = self.get_batch_gen(x_keys + y_keys,
                                       batch_size=batch_size,
                                       record_transform=record_transform, df=df)
        while True:
            batch = next(batch_gen)
            x = [batch[k] for k in x_keys]
            y = [batch[k] for k in y_keys]
            yield x, y

    def get_train_val_gen(self, x_keys, y_keys, batch_size=128,
                          record_transform=None, train_frac=.8):
        train_df = self.df.sample(frac=train_frac, random_state=200)
        val_df = self.df.drop(train_df.index)

        train_gen = self.get_train_gen(x_keys=x_keys, y_keys=y_keys,
                                       batch_size=batch_size,
                                       record_transform=record_transform,
                                       df=train_df)

        val_gen = self.get_train_gen(x_keys=x_keys, y_keys=y_keys,
                                     batch_size=batch_size,
                                     record_transform=record_transform,
                                     df=val_df)

        return train_gen, val_gen


class TubWriter(Tub):
    def __init__(self, *args, **kwargs):
        super(TubWriter, self).__init__(*args, **kwargs)

    def run(self, *args):
        """
        API function needed to use as a Donkey part. Accepts values,
        pairs them with their inputs keys and saves them to disk.
        """
        assert len(self.inputs) == len(args)
        record = dict(zip(self.inputs, args))
        self.put_record(record)
        return self.current_ix


class TubReader(Tub):
    def __init__(self, path, *args, **kwargs):
        super(TubReader, self).__init__(*args, **kwargs)

    def run(self, *args):
        """
        API function needed to use as a Donkey part.
        Accepts keys to read from the tub and retrieves them sequentially.
        """
        record_dict = self.get_record(self.read_ix)
        self.read_ix += 1
        record = [record_dict[key] for key in args]
        return record


class TubHandler:
    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def get_tub_list(self):
        folders = next(os.walk(self.path))[1]
        return folders

    def get_last_tub(self, folders=None):
        def get_tub_num(tub_name):
            try:
                tub_name_split = tub_name.split('_')
                num = int(tub_name_split[1])
                date_str = tub_name_split[-1]
            except:
                num = 0
                date_str = '00-00-00'
            return num, date_str

        if not folders:
            folders = self.get_tub_list()
        folders = sorted(folders)
        numbers = [get_tub_num(x)[0] for x in folders]
        last_number = max(numbers)
        last_tubs = [x for x in folders if get_tub_num(x)[0] == last_number]
        return last_number, last_tubs[-1]

    def next_tub_number(self):
        last_number, _ = self.get_last_tub()
        return  last_number + 1

    def create_tub_path(self):
        tub_num = self.next_tub_number()
        date = datetime.datetime.now().strftime('%y-%m-%d')
        name = '_'.join(['tub', str(tub_num), date])
        tub_path = os.path.join(self.path, name)
        return tub_path

    def new_tub_writer(self, inputs, types, user_meta=[], allow_reverse=True):
        tub_path = self.create_tub_path()
        tw = TubWriter(path=tub_path, inputs=inputs, types=types,
                       user_meta=user_meta, allow_reverse=allow_reverse)
        return tw


class TubImageStacker(Tub):
    """
    A Tub for training a NN with images that are the last three records stacked
    together as 3 channels of a single image. The idea is to give a simple
    feed-forward NN some chance of building a model based on motion. If you
    drive with the ImageFIFO part, then you don't need this. Just make sure
    your inference pass uses the ImageFIFO that the NN will now expect.
    """

    def rgb2gray(self, rgb):
        """
        take a numpy rgb image return a new single channel image converted to
        greyscale
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def stack3Images(self, img_a, img_b, img_c):
        """
        convert 3 rgb images into grayscale and put them into the 3 channels of
        a single output image
        """
        width, height, _ = img_a.shape

        gray_a = self.rgb2gray(img_a)
        gray_b = self.rgb2gray(img_b)
        gray_c = self.rgb2gray(img_c)

        img_arr = np.zeros([width, height, 3], dtype=np.dtype('B'))

        img_arr[..., 0] = np.reshape(gray_a, (width, height))
        img_arr[..., 1] = np.reshape(gray_b, (width, height))
        img_arr[..., 2] = np.reshape(gray_c, (width, height))

        return img_arr

    def get_record(self, ix):
        """
        get the current record and two previous.
        stack the 3 images into a single image.
        """

        data = super(TubImageStacker, self).get_record(ix)

        if ix > 1:
            data_ch1 = super(TubImageStacker, self).get_record(ix - 1)
            data_ch0 = super(TubImageStacker, self).get_record(ix - 2)

            json_data = self.get_json_record(ix)
            for key, val in json_data.items():
                typ = self.get_input_type(key)
                # load objects that were saved as separate files
                if typ == 'image':
                    val = self.stack3Images(data_ch0[key],
                                            data_ch1[key],
                                            data[key])
                    data[key] = val
                elif typ == 'image_array':
                    img = self.stack3Images(data_ch0[key],
                                            data_ch1[key],
                                            data[key])
                    val = np.array(img)
                    data[key] = val

        return data


class TubTimeStacker(TubImageStacker):
    '''
    A Tub for training N with records stacked through time.
    The idea here is to force the network to learn to look ahead in time.
    Init with an array of time offsets from the current time.
    '''

    def __init__(self, frame_list, *args, **kwargs):
        '''
        frame_list of [0, 10] would stack the current and 10 frames from now
        records togther in a single record with just the current image returned.
        [5, 90, 200] would return 3 frames of records, ofset 5, 90, and 200 f
        rames in the future.
        '''
        super(TubTimeStacker, self).__init__(*args, **kwargs)
        self.frame_list = frame_list

    def get_record(self, ix):
        '''
        stack the N records into a single record. Each key value has the record
        index with a suffix of _N where N is the frame offset into the data.
        '''
        data = {}
        for i, iOffset in enumerate(self.frame_list):
            iRec = ix + iOffset

            try:
                json_data = self.get_json_record(iRec)
            except FileNotFoundError:
                pass
            except:
                pass

            for key, val in json_data.items():
                typ = self.get_input_type(key)
                # load only the first image saved as separate files
                if typ == 'image' and i == 0:
                    val = Image.open(os.path.join(self.path, val))
                    data[key] = val
                elif typ == 'image_array' and i == 0:
                    d = super(TubTimeStacker, self).get_record(ix)
                    data[key] = d[key]
                else:
                    # we append a _offset to the key so user/angle out now be
                    # user/angle_0
                    new_key = key + "_" + str(iOffset)
                    data[new_key] = val
        return data


class TubGroup(Tub):
    def __init__(self, tub_paths):
        tub_paths = self.resolve_tub_paths(tub_paths)
        print('TubGroup:tubpaths:', tub_paths)
        tubs = [Tub(path) for path in tub_paths]
        self.input_types = {}

        record_count = 0
        for t in tubs:
            t.update_df()
            record_count += len(t.df)
            self.input_types.update(dict(zip(t.inputs, t.types)))

        print('joining the tubs {} records together. This could take {}'
              ' minutes.'.format(record_count, int(record_count / 300000)))

        self.meta = {'inputs': list(self.input_types.keys()),
                     'types': list(self.input_types.values())}

        self.df = pd.concat([t.df for t in tubs], axis=0, join='inner')

    def find_tub_paths(self, path):
        matches = []
        path = os.path.expanduser(path)
        for file in glob.glob(path):
            if os.path.isdir(file):
                matches.append(os.path.join(os.path.abspath(file)))
        return matches

    def resolve_tub_paths(self, path_list):
        path_list = path_list.split(",")
        resolved_paths = []
        for path in path_list:
            paths = self.find_tub_paths(path)
            resolved_paths += paths
        return resolved_paths

    def get_num_records(self):
        return len(self.df)


class TubWiper:
    """
    Donkey part which allows to delete a bunch of records from the end of tub.
    This allows to remove bad data already during recording. As this gets called
    in the vehicle loop the deletion runs only once in each continuous
    activation. A new execution requires to release of the input trigger. The
    action could result in a multiple number of executions otherwise.
    """
    def __init__(self, tub, num_records=20):
        """
        :param tub: tub to operate on
        :param num_records: number or records to delete
        """
        self._tub = tub
        self._num_records = num_records
        self._active_loop = False

    def run(self, is_delete):
        """
        Method in the vehicle loop. Delete records when trigger switches from
        False to True only.
        :param is_delete: if deletion has been triggered by the caller
        """
        # only run if input is true and debounced
        if is_delete:
            if not self._active_loop:
                # action command
                self._tub.remove_last_records(self._num_records)
                # increase the loop counter
                self._active_loop = True
        else:
            # trigger released, reset active loop
            self._active_loop = False

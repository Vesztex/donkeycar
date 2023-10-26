import datetime
import os
from threading import Thread
import json

from kivy import Logger
from kivy.clock import Clock
from kivy.properties import ObjectProperty, NumericProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from donkeycar.management.ui.common import FileChooserBase, tub_screen
from donkeycar.pipeline.database import PilotDatabase
from donkeycar.pipeline.training import train


class BackgroundColor(Widget):
    pass


class BackgroundLabel(BackgroundColor, Label):
    pass


class FittingLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_texture_size(self, o, new_size):
        super().on_texture_size(o, new_size)
        self.size = self.texture_size


class CustomButton(ToggleButton):
    menu = ObjectProperty()
    i = NumericProperty()


class CustomDropDown(DropDown):
    selected = ListProperty()

    def set_labels(self, labels):
        self.clear_widgets()
        self.selected.clear()
        print(f'len {len(self.selected)}')
        for i, label in enumerate(labels):
            but = CustomButton(i=i, text=label, menu=self)
            self.add_widget(but)


class TransferSelector(BoxLayout, FileChooserBase):
    """ Class to select transfer model"""
    filters = ['*.h5', '*.savedmodel']


class TrainScreen(Screen):
    """ Class showing the training screen. """
    config = ObjectProperty(force_dispatch=True, allownone=True)
    database = ObjectProperty()
    pilot_df = ObjectProperty(force_dispatch=True)
    tub_df = ObjectProperty(force_dispatch=True)
    train_checker = False

    def train_call(self, *args):
        tub_path = tub_screen().ids.tub_loader.tub.base_path
        transfer = self.ids.transfer_spinner.text
        model_type = self.ids.train_spinner.text
        if transfer != 'Choose transfer model':
            h5 = os.path.join(self.config.MODELS_PATH, transfer + '.h5')
            sm = os.path.join(self.config.MODELS_PATH, transfer + '.savedmodel')
            if os.path.exists(sm):
                transfer = sm
            elif os.path.exists(h5):
                transfer = h5
            else:
                transfer = None
                self.ids.status.text = \
                    f'Could find neither {sm} nor {h5} - training ' \
                    f'without transfer'
        else:
            transfer = None
        try:
            history = train(self.config, tub_paths=tub_path,
                            model_type=model_type,
                            transfer=transfer,
                            comment=self.ids.comment.text)
        except Exception as e:
            Logger.error(e)
            self.ids.status.text = f'Train failed see console'

    def train(self):
        self.config.SHOW_PLOT = False
        t = Thread(target=self.train_call)
        self.ids.status.text = 'Training started.'

        def func(dt):
            t.start()

        def check_training_done(dt):
            if not t.is_alive():
                self.train_checker.cancel()
                self.ids.comment.text = 'Comment'
                self.ids.transfer_spinner.text = 'Choose transfer model'
                self.ids.train_button.state = 'normal'
                self.ids.status.text = 'Training completed.'
                self.ids.train_button.disabled = False
                self.reload_database()

        # schedules the call after the current frame
        Clock.schedule_once(func, 0)
        # checks if training finished and updates the window if
        self.train_checker = Clock.schedule_interval(check_training_done, 0.5)

    def set_config_attribute(self, input):
        try:
            val = json.loads(input)
        except ValueError:
            val = input

        att = self.ids.cfg_spinner.text
        setattr(self.config, att, val)
        self.ids.status.text = f'Setting {att} to {val} of type ' \
                               f'{type(val).__name__}'

    def keys(self):
        if self.config:
            return [str(k) for k in self.config.__dict__.keys()]
        else:
            return []

    def on_config(self, obj, config):
        if self.config and self.ids:
            self.ids.cfg_spinner.values = self.keys()
            self.reload_database()

    def reload_database(self):
        if self.config:
            self.database = PilotDatabase(self.config)

    def on_database(self, obj=None, database=None):
        df = self.database.to_df()
        df.drop(columns=['History', 'Config'], errors='ignore', inplace=True)
        self.set_database_frame(df)

        pilot_names = self.database.get_pilot_names()
        self.ids.transfer_spinner.values \
            = ['Choose transfer model'] + pilot_names
        self.ids.select_spinner.values \
            = pilot_names

    def set_database_frame(self, df):
        grid = self.ids.scroll_pilots
        grid.clear_widgets()
        cols = len(df.columns)
        rows = len(df)
        grid.cols = cols

        for i, col in enumerate(df.columns):
            lab = BackgroundLabel(text=f"[b]{col}[/b]", markup=True)
            lab.size = lab.texture_size
            grid.add_widget(lab)
            if col in ('Pilot', 'Comment'):
                grid.cols_minimum |= {i: 100}

        for row in range(rows):
            for col in range(cols):
                cell = df.iloc[row][col]
                if df.columns[col] == 'Time':
                    cell = datetime.datetime.fromtimestamp(cell)
                    cell = cell.strftime("%Y-%m-%d %H:%M:%S")
                cell = str(cell)
                lab = BackgroundLabel(text=cell)
                lab.size = lab.texture_size
                grid.add_widget(lab)

        #self.ids.colomn_chooser.set_labels(df.columns)
        self.ids.column_chooser.text = ", ".join(df.columns)
import datetime
import os
from threading import Thread
import json

from kivy import Logger
from kivy.clock import Clock
from kivy.properties import ObjectProperty, NumericProperty, ListProperty, \
    StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget


from donkeycar.management.ui.common import FileChooserBase, get_app_screen, \
    AppScreen, status
from donkeycar.pipeline.database import PilotDatabase
from donkeycar.pipeline.training import train


class ConfigParamSetter(BoxLayout):
    screen = ObjectProperty()
    config = ObjectProperty(force_dispatch=True, allownone=True)
    button_text = StringProperty("+")
    is_global = False

    def get_keys(self):
        if self.config:
            return [str(k) for k in self.config.__dict__.keys()]
        else:
            return []

    def on_config(self, obj=None, config=None):
        if self.ids:
            self.ids.cfg_spinner.values = self.get_keys()

    def set_config_attribute(self, input):
        try:
            val = json.loads(input)
        except ValueError:
            val = input
        att = self.ids.cfg_spinner.text
        setattr(self.config, att, val)
        status(f'Setting {att} to {val} of type {type(val).__name__}')


class ConfigParamPanel(GridLayout):

    def row_count(self):
        rows = int(len(self.children) // self.cols) \
            + 0 if len(self.children) % self.cols == 0 else 1
        return rows

    def add(self):
        train_screen = get_app_screen('train')
        cfg_setter = ConfigParamSetter(screen=train_screen,
                                       config=train_screen.config,
                                       button_text='-')
        # We need simulate a config change so the keys get populated
        cfg_setter.on_config()
        self.add_widget(cfg_setter)


# class BackgroundColor(Widget):
#     pass


class BackgroundLabel(Label):
    pass


class MenuCheckbox(BoxLayout):
    menu = ObjectProperty()
    text = StringProperty()
    i = NumericProperty()


class CheckBoxRow(BoxLayout):
    selected = ListProperty()
    screen = ObjectProperty()

    def build_widgets(self, labels):
        self.clear_widgets()
        self.selected.clear()
        for i, label in enumerate(labels):
            but = MenuCheckbox(i=i, text=label, menu=self)
            self.add_widget(but)
        self.selected = list(labels)


class TransferSelector(BoxLayout, FileChooserBase):
    """ Class to select transfer model"""
    filters = ['*.h5', '*.savedmodel']


class TrainScreen(AppScreen):
    """ Class showing the training screen. """
    config = ObjectProperty(force_dispatch=True, allownone=True)
    database = ObjectProperty()
    dataframe = ObjectProperty(force_dispatch=True)
    pilot_df = ObjectProperty(force_dispatch=True)
    tub_df = ObjectProperty(force_dispatch=True)
    train_checker = False

    def train_call(self, *args):
        tub_path = get_app_screen('tub').ids.tub_loader.tub.base_path
        transfer = self.ids.transfer_spinner.text
        model_type = self.ids.train_spinner.text
        h5 = os.path.join(self.config.MODELS_PATH, transfer + '.h5')
        sm = os.path.join(self.config.MODELS_PATH, transfer + '.savedmodel')

        if transfer == 'Choose transfer model':
            transfer_model = None
        elif os.path.exists(sm):
            transfer_model = sm
        elif os.path.exists(h5):
            transfer_model = h5
        else:
            transfer_model = None
            status(f'Could find neither {sm} nor {h5} - training without '
                   f'transfer')
        try:
            history = train(self.config, tub_paths=tub_path,
                            model_type=model_type,
                            transfer=transfer_model,
                            comment=self.ids.comment.text)
        except Exception as e:
            Logger.error(e)
            status(f'Training failed see console')

    def train(self):
        self.config.SHOW_PLOT = False
        t = Thread(target=self.train_call)
        status('Training started.')

        def func(dt):
            t.start()

        def check_training_done(dt):
            if t.is_alive():
                return
            self.train_checker.cancel()
            self.ids.comment.text = 'Comment'
            self.ids.transfer_spinner.text = 'Choose transfer model'
            self.ids.train_button.state = 'normal'
            self.ids.train_button.disabled = False
            self.reload_database()
            status('Training completed.')

        # schedules the call after the current frame
        Clock.schedule_once(func, 0)
        # checks if training finished and updates the window if
        self.train_checker = Clock.schedule_interval(check_training_done, 0.5)

    def on_config(self, obj, config):
        if self.config and self.ids:
            self.reload_database()

    def reload_database(self):
        if self.config:
            self.database = PilotDatabase(self.config)

    def on_database(self, obj=None, database=None):
        df = self.database.to_df()
        df.drop(columns=['History', 'Config'], errors='ignore', inplace=True)
        self.dataframe = df

    def on_dataframe(self, obj, dataframe):
        self.plot_dataframe(dataframe)
        if self.dataframe.empty:
            return
        pilot_names = self.dataframe.loc[:, 'Name']
        self.ids.transfer_spinner.values \
            = ['Choose transfer model'] + pilot_names
        self.ids.select_spinner.values = pilot_names
        self.ids.column_chooser.build_widgets(dataframe)

    def plot_dataframe(self, df, selected_cols=None):
        grid = self.ids.scroll_pilots
        grid.clear_widgets()
        # only set column chooser labels on initialisation when selected_cols
        # is not passed in. otherwise project df to selected columns
        if selected_cols is not None:
            df = df[selected_cols]

        num_cols = len(df.columns)
        rows = len(df)
        grid.cols = num_cols

        for i, col in enumerate(df.columns):
            lab = BackgroundLabel(text=f"[b]{col}[/b]", markup=True)
            lab.size = lab.texture_size
            grid.add_widget(lab)
            if col in ('Pilot', 'Comment'):
                grid.cols_minimum |= {i: 100}

        for row in range(rows):
            for col in range(num_cols):
                cell = df.iloc[row][col]
                if df.columns[col] == 'Time':
                    cell = datetime.datetime.fromtimestamp(cell)
                    cell = cell.strftime("%Y-%m-%d %H:%M:%S")
                cell = str(cell)
                lab = BackgroundLabel(text=cell)
                lab.size = lab.texture_size
                grid.add_widget(lab)

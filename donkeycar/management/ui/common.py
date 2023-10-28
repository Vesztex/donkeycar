import os
import io
import time

import numpy as np
from functools import partial
from PIL import Image as PilImage

from kivy import Logger
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty, ListProperty, \
    BooleanProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage

from donkeycar.management.ui.rc_file_handler import rc_handler

LABEL_SPINNER_TEXT = 'Add/remove'


def tub_screen():
    return App.get_running_app().tub_screen if App.get_running_app() else None


def pilot_screen():
    return App.get_running_app().pilot_screen if App.get_running_app() else None


def train_screen():
    return App.get_running_app().train_screen if App.get_running_app() else None


def car_screen():
    return App.get_running_app().car_screen if App.get_running_app() else None


def decompose(field):
    """ Function to decompose a string vector field like 'gyroscope_1' into a
        tuple ('gyroscope', 1) """
    field_split = field.split('_')
    if len(field_split) > 1 and field_split[-1].isdigit():
        return '_'.join(field_split[:-1]), int(field_split[-1])
    return field, None


def get_norm_value(value, cfg, field_property, normalised=True):
    max_val_key = field_property.max_value_id
    max_value = getattr(cfg, max_val_key, 1.0)
    out_val = value / max_value if normalised else value * max_value
    return out_val


class MySpinnerOption(SpinnerOption):
    """ Customization for Spinner """
    pass


class MySpinner(Spinner):
    """ Customization of Spinner drop down menu """
    def __init__(self, **kwargs):
        super().__init__(option_cls=MySpinnerOption, **kwargs)


class FileChooserPopup(Popup):
    """ File Chooser popup window"""
    load = ObjectProperty()
    root_path = StringProperty()
    filters = ListProperty()


class FileChooserBase:
    """ Base class for file chooser widgets"""
    file_path = StringProperty("No file chosen")
    popup = ObjectProperty(None)
    root_path = os.path.expanduser('~')
    title = StringProperty(None)
    filters = ListProperty()

    def open_popup(self):
        self.popup = FileChooserPopup(load=self.load, root_path=self.root_path,
                                      title=self.title, filters=self.filters)
        self.popup.open()

    def load(self, selection):
        """ Method to load the chosen file into the path and call an action"""
        self.file_path = str(selection[0])
        self.popup.dismiss()
        self.load_action()

    def load_action(self):
        """ Virtual method to run when file_path has been updated """
        pass


class LabelBar(BoxLayout):
    """ Widget that combines a label with a progress bar. This is used to
        display the record fields in the data panel."""
    field = StringProperty()
    field_property = ObjectProperty()
    config = ObjectProperty()
    msg = ''

    def update(self, record):
        """ This function is called everytime the current record is updated"""
        if not record:
            return
        field, index = decompose(self.field)
        if field in record.underlying:
            val = record.underlying[field]
            if index is not None:
                val = val[index]
            # Update bar if a field property for this field is known
            if self.field_property:
                norm_value = get_norm_value(val, self.config,
                                            self.field_property)
                new_bar_val = (norm_value + 1) * 50 if \
                    self.field_property.centered else norm_value * 100
                self.ids.bar.value = new_bar_val
            self.ids.field_label.text = self.field
            if isinstance(val, float) or isinstance(val, np.float32):
                text = f'{val:+07.3f}'
            elif isinstance(val, int):
                text = f'{val:10}'
            else:
                text = str(val)
            self.ids.value_label.text = text
        else:
            Logger.error(f'Record: Bad record {record.underlying["_index"]} - '
                         f'missing field {self.field}')


class DataPanel(BoxLayout):
    """ Data panel widget that contains the label/bar widgets and the drop
        down menu to select/deselect fields."""
    record = ObjectProperty()
    # dual mode is used in the pilot arena where we only show angle and
    # throttle or speed
    dual_mode = BooleanProperty(False)
    auto_text = StringProperty(LABEL_SPINNER_TEXT)
    throttle_field = StringProperty('user/throttle')
    link = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = {}
        self.screen = ObjectProperty()

    def add_remove(self):
        """ Method to add or remove a LabelBar. Depending on the value of the
            drop down menu the LabelBar is added if it is not present otherwise
            removed."""
        field = self.ids.data_spinner.text
        if field is LABEL_SPINNER_TEXT:
            return
        if field in self.labels and not self.dual_mode:
            self.remove_widget(self.labels[field])
            del self.labels[field]
            self.screen.status(f'Removing {field}')
        else:
            # in dual mode replace the second entry with the new one
            if self.dual_mode and len(self.labels) == 2:
                k, v = list(self.labels.items())[-1]
                self.remove_widget(v)
                del self.labels[k]
            field_property = rc_handler.field_properties.get(decompose(field)[0])
            cfg = tub_screen().ids.config_manager.config
            lb = LabelBar(field=field, field_property=field_property, config=cfg)
            self.labels[field] = lb
            self.add_widget(lb)
            lb.update(self.record)
            if len(self.labels) == 2:
                self.throttle_field = field
            self.screen.status(f'Adding {field}')
        if self.screen.name == 'tub':
            self.screen.ids.data_plot.plot_from_current_bars()
        self.ids.data_spinner.text = LABEL_SPINNER_TEXT
        self.auto_text = field

    def on_record(self, obj, record):
        """ Kivy function that is called every time self.record changes"""
        for v in self.labels.values():
            v.update(record)

    def clear(self):
        for v in self.labels.values():
            self.remove_widget(v)
        self.labels.clear()


class FullImage(Image):
    """ Widget to display an image that fills the space. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core_image = None

    def update(self, record):
        """ This method is called ever time a record gets updated. """
        try:
            img_arr = self.get_image(record)
            pil_image = PilImage.fromarray(img_arr)
            bytes_io = io.BytesIO()
            pil_image.save(bytes_io, format='png')
            bytes_io.seek(0)
            self.core_image = CoreImage(bytes_io, ext='png')
            self.texture = self.core_image.texture
        except KeyError as e:
            Logger.error(f'Record: Missing key: {e}')
        except Exception as e:
            Logger.error(f'Record: Bad record: {e}')

    def get_image(self, record):
        return record.image()


class ControlPanel(BoxLayout):
    """ Class for control panel navigation. """
    screen = ObjectProperty()
    speed = NumericProperty(1.0)
    record_display = StringProperty()
    clock = None
    fwd = None

    def start(self, fwd=True, continuous=False):
        """
        Method to cycle through records if either single <,> or continuous
        <<, >> buttons are pressed
        :param fwd:         If we go forward or backward
        :param continuous:  If we do <<, >> or <, >
        :return:            None
        """
        # this widget it used in two screens, so reference the original location
        # of the config which is the config manager in the tub screen
        cfg = tub_screen().ids.config_manager.config
        hz = cfg.DRIVE_LOOP_HZ if cfg else 20
        time.sleep(0.1)
        call = partial(self.step, fwd, continuous)
        if continuous:
            self.fwd = fwd
            s = float(self.speed) * hz
            cycle_time = 1.0 / s
        else:
            cycle_time = 0.08
        self.clock = Clock.schedule_interval(call, cycle_time)

    def step(self, fwd=True, continuous=False, *largs):
        """
        Updating a single step and cap/floor the index so we stay w/in the tub.
        :param fwd:         If we go forward or backward
        :param continuous:  If we are in continuous mode <<, >>
        :param largs:       dummy
        :return:            None
        """
        if self.screen.index is None:
            self.screen.status("No tub loaded")
            return
        new_index = self.screen.index + (1 if fwd else -1)
        if new_index >= tub_screen().ids.tub_loader.len:
            new_index = 0
        elif new_index < 0:
            new_index = tub_screen().ids.tub_loader.len - 1
        self.screen.index = new_index
        msg = f'Donkey {"run" if continuous else "step"} ' \
              f'{"forward" if fwd else "backward"}'
        if not continuous:
            msg += f' - you can also use {"<right>" if fwd else "<left>"} key'
        else:
            msg += ' - you can toggle run/stop with <space>'
        self.screen.status(msg)

    def stop(self):
        if self.clock:
            self.clock.cancel()
            self.clock = None

    def restart(self):
        if self.clock:
            self.stop()
            self.start(self.fwd, True)

    def update_speed(self, up=True):
        """ Method to update the speed on the controller"""
        values = self.ids.control_spinner.values
        idx = values.index(self.ids.control_spinner.text)
        if up and idx < len(values) - 1:
            self.ids.control_spinner.text = values[idx + 1]
        elif not up and idx > 0:
            self.ids.control_spinner.text = values[idx - 1]

    def set_button_status(self, disabled=True):
        """ Method to disable(enable) all buttons. """
        self.ids.run_bwd.disabled = self.ids.run_fwd.disabled = \
            self.ids.step_fwd.disabled = self.ids.step_bwd.disabled = disabled

    def on_keyboard(self, key, scancode):
        """ Method to chack with keystroke has ben sent. """
        if key == ' ':
            if self.clock and self.clock.is_triggered:
                self.stop()
                self.set_button_status(disabled=False)
                self.screen.status('Donkey stopped')
            else:
                self.start(continuous=True)
                self.set_button_status(disabled=True)
        elif scancode == 79:
            self.step(fwd=True)
        elif scancode == 80:
            self.step(fwd=False)
        elif scancode == 45:
            self.update_speed(up=False)
        elif scancode == 46:
            self.update_speed(up=True)


class PaddedBoxLayout(BoxLayout):
    pass


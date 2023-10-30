from kivy.logger import Logger, LOG_LEVELS

import os
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen

from donkeycar.management.ui.car_screen import CarScreen
from donkeycar.management.ui.pilot_screen import PilotScreen
from donkeycar.management.ui.train_screen import TrainScreen
from donkeycar.management.ui.tub_screen import TubScreen

Logger.setLevel(LOG_LEVELS["info"])

Builder.load_file(os.path.join(os.path.dirname(__file__), 'ui.kv'))
Window.clearcolor = (0.2, 0.2, 0.2, 1)


class TabBar(BoxLayout):
    manager = ObjectProperty(None)

    def disable_only(self, bar_name):
        this_button_name = bar_name + '_btn'
        for button_name, button in self.ids.items():
            button.disabled = button_name == this_button_name


class StartScreen(Screen):
    img_path = os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        '../../parts/web_controller/templates/'
        'static/donkeycar-logo-sideways.png'))
    pass


class DonkeyApp(App):
    start_screen = None
    tub_screen = None
    train_screen = None
    pilot_screen = None
    car_screen = None
    title = 'Donkey Manager'

    def initialise(self, event):
        self.tub_screen.ids.config_manager.load_action()
        self.pilot_screen.initialise(event)
        self.car_screen.initialise()
        # This builds the graph which can only happen after everything else
        # has run, therefore delay until the next round.
        Clock.schedule_once(self.tub_screen.ids.tub_loader.update_tub)

    def build(self):
        self.start_screen = StartScreen(name='donkey')
        self.tub_screen = TubScreen(name='tub')
        self.train_screen = TrainScreen(name='train')
        self.pilot_screen = PilotScreen(name='pilot')
        self.car_screen = CarScreen(name='car')
        Window.bind(on_keyboard=self.tub_screen.on_keyboard)
        Window.bind(on_keyboard=self.pilot_screen.on_keyboard)
        Window.bind(on_request_close=self.on_request_close)
        Clock.schedule_once(self.initialise)
        sm = ScreenManager()
        sm.add_widget(self.start_screen)
        sm.add_widget(self.tub_screen)
        sm.add_widget(self.train_screen)
        sm.add_widget(self.pilot_screen)
        sm.add_widget(self.car_screen)
        return sm

    def on_request_close(self, *args):
        tub = self.tub_screen.ids.tub_loader.tub
        if tub:
            tub.close()
        Logger.info("Good bye Donkey")
        return False


def main():
    tub_app = DonkeyApp()
    tub_app.run()


if __name__ == '__main__':
    main()

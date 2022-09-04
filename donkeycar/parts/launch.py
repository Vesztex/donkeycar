import time
import logging
from donkeycar.parts import Part

logger = logging.getLogger(__name__)


class AiLaunch(Part):
    '''
    This part will apply a large thrust on initial activation. This is to help
    in racing to start fast and then the ai will take over quickly when it's
    up to speed.
    '''

    def __init__(self, launch_duration=1.0, launch_throttle=1.0,
                 keep_enabled=False):
        """
        Creation of the AiLaunch part
        :param float launch_duration:   How long to give launch throttle in
                                        seconds (default 1.0)
        :param float launch_throttle:   How much launch throttle (default 1.0)
        :param bool keep_enabled:       Keeping it enabled after launch (
                                        default False)
        """
        # always call Part constructor to make arguments available in
        # documentation
        super().__init__(launch_duration=launch_duration,
                         launch_throttle=launch_throttle,
                         keep_enabled=keep_enabled)
        self.active = False
        self.enabled = False
        self.timer_start = None
        self.timer_duration = launch_duration
        self.launch_throttle = launch_throttle
        self.prev_mode = None
        self.trigger_on_switch = keep_enabled
        
    def enable_ai_launch(self):
        self.enabled = True
        logger.info('AiLauncher is enabled.')

    def run(self, mode, ai_throttle):
        new_throttle = ai_throttle

        if mode != self.prev_mode:
            self.prev_mode = mode
            if mode == "local" and self.trigger_on_switch:
                self.enabled = True

        if mode == "local" and self.enabled:
            if not self.active:
                self.active = True
                self.timer_start = time.time()
            else:
                duration = time.time() - self.timer_start
                if duration > self.timer_duration:
                    self.active = False
                    self.enabled = False
        else:
            self.active = False

        if self.active:
            logger.info('AiLauncher is active!!!')
            new_throttle = self.launch_throttle

        return new_throttle


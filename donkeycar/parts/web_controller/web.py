#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:10:44 2017
@author: wroscoe
remotes.py
The client and web server needed to control a car remotely.
"""


import os
import json
import logging
import time
import asyncio

import requests
from tornado.ioloop import IOLoop
from tornado.web import Application, RedirectHandler, StaticFileHandler, \
    RequestHandler
from tornado.httpserver import HTTPServer
import tornado.gen
import tornado.websocket
from socket import gethostname

from .. import Part, PartType
from ... import utils

logger = logging.getLogger(__name__)


class RemoteWebServer:
    '''
    A controller that repeatedly polls a remote webserver and expects
    the response to be angle, throttle and drive mode.
    '''

    def __init__(self, remote_url, connection_timeout=.25):

        self.control_url = remote_url
        self.time = 0.
        self.angle = 0.
        self.throttle = 0.
        self.mode = 'user'
        self.mode_latch = None
        self.recording = False
        # use one session for all requests
        self.session = requests.Session()

    def update(self):
        '''
        Loop to run in separate thread the updates angle, throttle and
        drive mode.
        '''

        while True:
            # get latest value from server
            self.angle, self.throttle, self.mode, self.recording = self.run()

    def run_threaded(self):
        '''
        Return the last state given from the remote server.
        '''
        return self.angle, self.throttle, self.mode, self.recording

    def run(self):
        '''
        Posts current car sensor data to webserver and returns
        angle and throttle recommendations.
        '''

        data = {}
        response = None
        while response is None:
            try:
                response = self.session.post(self.control_url,
                                             files={'json': json.dumps(data)},
                                             timeout=0.25)

            except requests.exceptions.ReadTimeout as err:
                print("\n Request took too long. Retrying")
                # Lower throttle to prevent runaways.
                return self.angle, self.throttle * .8, None

            except requests.ConnectionError as err:
                # try to reconnect every 3 seconds
                print("\n Vehicle could not connect to server. Make sure you've " +
                    "started your server and you're referencing the right port.")
                time.sleep(3)

        data = json.loads(response.text)
        angle = float(data['angle'])
        throttle = float(data['throttle'])
        drive_mode = str(data['drive_mode'])
        recording = bool(data['recording'])

        return angle, throttle, drive_mode, recording

    def shutdown(self):
        pass


class LocalWebController(Application, Part):
    """ Part that runs a local web server on the car (or wherever the donkey
    car software is started on). Connecting to the web server through your
    browser allows you to view the current camera image and to use a game
    controller or the web page embedded track pad to drive the car. The web
    controller allows to switch between different modes, like driving with
    the controller, or with the autopilot or in mixed mode, with steering
    from autopilot and user controlled throttle. It also allows to start/stop
    the recording.
    """

    def __init__(self, port=8887, mode='user'):
        """
        Create and publish variables needed on many of
        the web handlers.
        """
        Part.__init__(self, port=port, mode=mode)
        logger.info('Starting Donkey Server...')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = mode
        self.mode_latch = None
        self.recording = False
        self.recording_latch = None
        self.config = {}
        self.buttons = {}  # latched button values for processing
        self.port = port
        self.num_records = 0
        self.wsclients = []
        self.loop = None

        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/wsDrive", WebSocketDriveAPI),
            (r"/wsCalibrate", WebSocketCalibrateAPI),
            (r"/calibrate", CalibrateHandler),
            (r"/video", VideoAPI),
            (r"/wsTest", WsTest),

            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]

        settings = {'debug': True}
        Application.__init__(self, handlers, **settings)
        logger.info(f"... you can now go to {gethostname()}.local:{port} to "
                    f"drive your car.")

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = tornado.ioloop.IOLoop.current()
        self.listen(self.port)
        self.loop.start()

    def update_wsclients(self, data):
        if data:
            for wsclient in self.wsclients:
                try:
                    data_str = json.dumps(data)
                    logger.debug(f"Updating web client: {data_str}")
                    wsclient.write_message(data_str)
                except Exception as e:
                    logger.warn("Error writing websocket message", exc_info=e)
                    pass

    def run_threaded(self, img_arr=None, num_records=0,
                     mode=None, recording=None):
        """
        :param np.array img_arr:    Current camera image or None
        :param int num_records:     Current number of data records
        :param str mode:            Default mode, 'user' or 'local angle' or
                                    'local pilot' - optional parameter
        :param bool recording:      Default recording mode - optional parameter
        :return tuple:              Returns a tuple of angle (float),
                                    throttle (float), mode (str),
                                    recording (bool) and controller_config (
                                    dict). The last one is only non-None if we
                                    are running the calibration.
        """
        self.img_arr = img_arr
        self.num_records = num_records
        #
        # enforce defaults if they are not none.
        #
        changes = {}
        if mode is not None and self.mode != mode:
            self.mode = mode
            changes["driveMode"] = self.mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
            changes["driveMode"] = self.mode
        if recording is not None and self.recording != recording:
            self.recording = recording
            changes["recording"] = self.recording
        if self.recording_latch is not None:
            self.recording = self.recording_latch
            self.recording_latch = None
            changes["recording"] = self.recording

        # Send record count to websocket clients
        if self.num_records is not None and self.recording is True:
            if self.num_records % 10 == 0:
                changes['num_records'] = self.num_records

        #
        # get latched button presses then clear button presses
        # Next iteration will clear press in memory
        #
        buttons = self.buttons
        self.buttons = {}
        for button, pressed in buttons.items():
            if pressed:
                self.buttons[button] = False

        # if there were changes, then send to web client
        if changes and self.loop is not None:
            logger.debug(str(changes))
            self.loop.add_callback(lambda: self.update_wsclients(changes))

        return self.angle, self.throttle, self.mode, \
            self.recording, buttons, self.config

    def shutdown(self):
        self.loop.stop()

    @classmethod
    def create(cls, cfg, port=None, mode=None):
        """
        Creation of the WebController using the config file.

        :param Config cfg:  donkey config object
        :param int port:    port number, defaults to config parameter
                            WEB_CONTROL_PORT if not overwritten
        :param str mode:    web controller start up mode, defaults to config
                            parameter WEB_INIT_MODE if not overwritten
        :return:            web controller instance
        """
        port = port or cfg.WEB_CONTROL_PORT
        mode = mode or cfg.WEB_INIT_MODE
        return LocalWebController(port=port, mode=mode)


class DriveAPI(RequestHandler):

    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)

    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)

        if data.get('angle') is not None:
            self.application.angle = data['angle']
        if data.get('throttle') is not None:
            self.application.throttle = data['throttle']
        if data.get('drive_mode') is not None:
            self.application.mode = data['drive_mode']
        if data.get('recording') is not None:
            self.application.recording = data['recording']
        if data.get('buttons') is not None:
            latch_buttons(self.application.buttons, data['buttons'])


class WsTest(RequestHandler):
    def get(self):
        data = {}
        self.render("templates/wsTest.html", **data)


class CalibrateHandler(RequestHandler):
    """ Serves the calibration web page"""
    async def get(self):
        await self.render("templates/calibrate.html")


def latch_buttons(buttons, pushes):
    """
    Latch button pushes
    buttons: the latched values
    pushes: the update value
    """
    if pushes is not None:
        #
        # we got button pushes.
        # - we latch the pushed buttons so we can process the push
        # - after it is processed we clear it
        #
        for button in pushes:
            # if pushed, then latch it
            if pushes[button]:
                buttons[button] = True


class WebSocketDriveAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("New client connected")
        self.application.wsclients.append(self)

    def on_message(self, message):
        data = json.loads(message)
        self.application.angle = data.get('angle', self.application.angle)
        self.application.throttle = data.get('throttle', self.application.throttle)
        if data.get('drive_mode') is not None:
            self.application.mode = data['drive_mode']
            self.application.mode_latch = self.application.mode
        if data.get('recording') is not None:
            self.application.recording = data['recording']
            self.application.recording_latch = self.application.recording
        if data.get('buttons') is not None:
            latch_buttons(self.application.buttons, data['buttons'])

    def on_close(self):
        # print("Client disconnected")
        self.application.wsclients.remove(self)


class WebSocketCalibrateAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("New client connected")

    def on_message(self, message):
        print(f"wsCalibrate {message}")
        data = json.loads(message)
        if 'throttle' in data:
            print(data['throttle'])
            self.application.throttle = data['throttle']

        if 'angle' in data:
            print(data['angle'])
            self.application.angle = data['angle']

        if 'config' in data:
            config = data['config']
            self.application.config = config

    def on_close(self):
        print("Client disconnected")


class VideoAPI(RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle.
    '''

    async def get(self):

        self.set_header("Content-type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:

            interval = .001
            if served_image_timestamp + interval < time.time() and \
                    hasattr(self.application, 'img_arr'):

                img = utils.arr_to_binary(self.application.img_arr)
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                served_image_timestamp = time.time()
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass
            else:
                await tornado.gen.sleep(interval)


class BaseHandler(RequestHandler):
    """ Serves the FPV web page"""
    async def get(self):
        data = {}
        await self.render("templates/base_fpv.html", **data)


class WebFpv(Application, Part):
    """
    Class for running an FPV web server that only shows the camera in real-time.
    The web page contains the camera view and auto-adjusts to the web browser
    window size. Conjecture: this picture up-scaling is performed by the
    client OS using graphics acceleration. Hence a web browser on the PC is
    faster than a pure python application based on open cv or similar.
    """
    part_type = PartType.PROCESS

    def __init__(self, port=8890):
        """
        Creates the WebFpv part. This starts a webserver with a single page
        containing the camera image of the car and nothing else.

        :param int port: Optional port number, defaults to 8890 if not given.
        """
        Part.__init__(self, port=port)
        self.port = port
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')

        """Construct and serve the tornado application."""
        handlers = [
            (r"/", BaseHandler),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path})
        ]

        settings = {'debug': True}
        Application.__init__(self, handlers, **settings)
        logger.info(f"Started Web FPV server. You can now go to "
                    f"{gethostname()}.local:{self.port} to view the car camera")
        self.loop = None
        self.img_arr = None

    def update(self):
        """ Thread interface for donkey car. Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = tornado.ioloop.IOLoop.current()
        self.listen(self.port)
        self.loop.start()

    def run_threaded(self, img_arr=None):
        """
        Donkeycar interface for threaded parts. The parts reads an image
        array and displays is on the webservers endpoint.

        :param np.array img_arr:    camera image to be displayed
        """
        self.img_arr = img_arr

    def shutdown(self):
        self.loop.stop()

    @classmethod
    def create(cls, cfg, port=None):
        port = port or cfg.WEB_FPV_PORT
        return cls(port=port)



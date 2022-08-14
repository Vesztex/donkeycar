from tornado import testing
import tornado.websocket
import tornado.web
import tornado.ioloop
import json
from donkeycar.parts.web_controller.web import WebSocketCalibrateAPI


class WebSocketCalibrateTest(testing.AsyncHTTPTestCase):
    """
    Example of WebSocket usage as a client in AsyncHTTPTestCase-based unit
    tests.
    """

    def get_app(self):
        app = tornado.web.Application([('/', WebSocketCalibrateAPI)])
        self.app = app
        return app

    def get_ws_url(self):
        return "ws://localhost:" + str(self.get_http_port()) + "/"

    @tornado.testing.gen_test
    def test_calibrate_servo_esc_1b(self):
        ws_client = yield tornado.websocket.websocket_connect(self.get_ws_url())
        data = {"config": {"STEERING_LEFT_PWM": 444}}
        yield ws_client.write_message(json.dumps(data))
        yield ws_client.close()
        pwm = self.app.config.get("STEERING_LEFT_PWM")
        assert pwm == 444

    @tornado.testing.gen_test
    def test_calibrate_servo_esc_2b(self):
        ws_client = yield tornado.websocket.websocket_connect(self.get_ws_url())
        data = {"config": {"STEERING_RIGHT_PWM": 555}}
        yield ws_client.write_message(json.dumps(data))
        yield ws_client.close()
        pwm = self.app.config.get("STEERING_RIGHT_PWM")
        assert pwm == 555

    @tornado.testing.gen_test
    def test_calibrate_servo_esc_3b(self):
        ws_client = yield tornado.websocket.websocket_connect(self.get_ws_url())
        data = {"config": {"THROTTLE_FORWARD_PWM": 666}}
        yield ws_client.write_message(json.dumps(data))
        yield ws_client.close()
        pwm = self.app.config.get("THROTTLE_FORWARD_PWM")
        assert pwm == 666

    @tornado.testing.gen_test
    def test_calibrate_mm1(self):
        ws_client = yield tornado.websocket.websocket_connect(self.get_ws_url())
        data = {"config": {"MM1_STEERING_MID": 1234}}
        yield ws_client.write_message(json.dumps(data))
        yield ws_client.close()
        pwm = self.app.config.get("MM1_STEERING_MID")
        assert pwm == 1234


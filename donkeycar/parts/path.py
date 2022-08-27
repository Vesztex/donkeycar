import pickle
import math
import logging
import os

import numpy
from PIL import Image, ImageDraw

from donkeycar.parts import PartType, Part
from donkeycar.utils import norm_deg, dist, deg2rad, arr_to_img

logger = logging.getLogger(__name__)


class Path(object):
    def __init__(self, min_dist=1., file_name="donkey_path.pkl"):
        self.path = []
        self.min_dist = min_dist
        self.x = math.inf
        self.y = math.inf
        self.recording = True
        self.file_name = file_name
        if os.path.exists(self.file_name):
            self.load()
            print("#" * 79)
            print("Loaded path:", self.file_name)
            print("Make sure your car is sitting at the origin of the path.")
            print("View web page and refresh. You should see your path.")
            print("Hit 'select' twice to change to ai drive mode.")
            print("You can press the X button (e-stop) to stop the car at any time.")
            print("Delete file", self.file_name, "and re-start")
            print("to record a new path.")
            print("#" * 79)

        else:
            print("#" * 79)
            print("You are now in record mode. Open the web page to your car")
            print("and as you drive you should see a path.")
            print("Complete one circuit of your course.")
            print("When you have exactly looped, or just shy of the ")
            print("loop, then save the path (press Circle).")
            print("You can also erase a path with the Triangle button.")
            print("When you're done, close this process with Ctrl+C.")
            print("Place car exactly at the start.")
            print("Then restart the car with 'python manage drive'.")
            print("It will reload the path and you will be ready to  ")
            print("follow the path using  'select' to change to ai drive mode.")
            print("You can also press the Square button to reset the origin")
            print("#" * 79)

    def run(self, x, y, save_path=False, erase_path=False):
        d = dist(x, y, self.x, self.y)
        if self.recording and d > self.min_dist:
            self.path.append((x, y))
            logging.info(f"path point ({x}, {y})")
            self.x = x
            self.y = y
        result = self.path
        # save or erase path if corresponding button is pressed.
        if save_path:
            self.save()
        if erase_path:
            self.erase()
        return result

    def save(self):
        logger.info(f"Saving path {self.file_name}")
        outfile = open(self.file_name, 'wb')
        pickle.dump(self.path, outfile)
    
    def load(self):
        logger.info(f"Loading path {self.file_name}")
        infile = open(self.file_name, 'rb')
        self.path = pickle.load(infile)
        self.recording = False

    def erase(self):
        if os.path.exists(self.file_name):
            logger.info(f"Removing path {self.file_name}")
            os.remove(self.file_name)
        else:
            logger.warning("No path found to erase")


class PImage(Part):
    def __init__(self, resolution=(500, 500), color="white",
                 clear_each_frame=False):
        """
        Creating the PImage part. This produces a single fixed image every
        time.

        :param tuple resolution:        Number of pixels WxH
        :param str color:               Color name, like 'white'
        :param bool clear_each_frame:   If a new image is created every time
        """
        super().__init__(resolution=resolution, color=color,
                         clear_each_frame=clear_each_frame)
        self.resolution = resolution
        self.color = color
        self.img = Image.new('RGB', resolution, color=color)
        self.clear_each_frame = clear_each_frame

    def run(self):
        """
        Donkey car part interface. Returns an image of a fixed color and
        resolution.

        :return:    Image
        :rtype:     PIL Image
        """
        if self.clear_each_frame:
            self.img = Image.new('RGB', self.resolution, color=self.color)

        return self.img


class OriginOffset(Part):
    """
    Use this part to set the car back to the origin without restarting it.
    """
    part_type = PartType.PLAN

    def __init__(self):
        """
        Creation of the OriginOffset part. Requires the joystick button name
        to trigger a reset of the origin.
        """
        super().__init__()
        self.ox = 0.0
        self.oy = 0.0
        self.last_x = 0.
        self.last_y = 0.

    def run(self, x, y, reset_origin=False):
        """
        Donkey Car parts interface. Saves x, y position values and returns x,
        y shifted bye the origin values. If the button_down list contains the
        init_button from the constructor, the origin is reset to the latest
        coordinates.

        :param float x:                 Input x coordinate
        :param float y:                 Input y coordinate
        :param bool reset_origin:       If the origin should be reset to
                                        the last position.

        :return:                        Output x,y coordinates
        """
        self.last_x = x
        self.last_y = y
        pos = x + self.ox, y + self.oy
        if reset_origin is True:
            self.init_to_last()
        return pos

    def init_to_last(self):
        self.ox = -self.last_x
        self.oy = -self.last_y


class PathPlot(Part):
    '''
    Part that draws a path plot on to an image
    '''
    def __init__(self, scale=1.0, offset=(0., 0.0)):
        """
        Creating the PathPlot part.

        :param scale:   Scale of the path
        :param offset:  Offset of the path
        """
        super().__init__(scale=scale, offset=offset)
        self.scale = scale
        self.offset = offset

    def plot_line(self, sx, sy, ex, ey, draw, color):
        """
        scale dist so that max_dist is edge of img (mm)
        and img is PIL Image, draw the line using the draw ImageDraw object
        """
        draw.line((sx, sy, ex, ey), fill=color, width=1)

    def run(self, img, path):
        
        if type(img) is numpy.ndarray:
            stacked_img = numpy.stack((img,)*3, axis=-1)
            img = arr_to_img(stacked_img)

        draw = ImageDraw.Draw(img)
        color = (255, 0, 0)
        for iP in range(0, len(path) - 1):
            ax, ay = path[iP]
            bx, by = path[iP + 1]
            self.plot_line(ax * self.scale + self.offset[0],
                        ay * self.scale + self.offset[1], 
                        bx * self.scale + self.offset[0], 
                        by * self.scale + self.offset[1], 
                        draw, 
                        color)

        return img


class PlotCircle(object):
    '''
    draw a circle plot to an image
    '''
    def __init__(self,  scale=1.0, offset=(0., 0.0), radius=4, color = (0, 255, 0)):
        self.scale = scale
        self.offset = offset
        self.radius = radius
        self.color = color

    def plot_circle(self, x, y, rad, draw, color, width=1):
        '''
        scale dist so that max_dist is edge of img (mm)
        and img is PIL Image, draw the circle using the draw ImageDraw object
        '''
        sx = x - rad
        sy = y - rad
        ex = x + rad
        ey = y + rad

        draw.ellipse([(sx, sy), (ex, ey)], fill=color)


    def run(self, img, x, y):
        draw = ImageDraw.Draw(img)
        self.plot_circle(x * self.scale + self.offset[0],
                        y * self.scale + self.offset[1], 
                        self.radius,
                        draw, 
                        self.color)

        return img


from donkeycar.la import Line3D, Vec3


class CTE(Part):
    """ Part that measures the cross track error"""
    def __int__(self):
        """
        Creating the CTE part
        :return:
        """
        super().__init__()

    def nearest_two_pts(self, path, x, y):
        if len(path) < 2:
            return None, None

        distances = []
        for iP, p in enumerate(path):
            d = dist(p[0], p[1], x, y)
            distances.append((d, iP, p))
        distances.sort(key=lambda elem: elem[0])
        iA = (distances[0][1] - 1) % len(path)
        a = path[iA]
        # iB is the next element in the path, wrapping around..
        iB = (iA + 2) % len(path)
        b = path[iB]
        
        return a, b

    def run(self, path, x, y):
        cte = 0.

        a, b = self.nearest_two_pts(path, x, y)
        
        if a and b:
            #logging.info("nearest: (%f, %f) to (%f, %f)" % ( a[0], a[1], x, y))
            a_v = Vec3(a[0], 0., a[1])
            b_v = Vec3(b[0], 0., b[1])
            p_v = Vec3(x, 0., y)
            line = Line3D(a_v, b_v)
            err = line.vector_to(p_v)
            sign = 1.0
            cp = line.dir.cross(err.normalized())
            if cp.y > 0.0 :
                sign = -1.0
            cte = err.mag() * sign            

        return cte


class PID_Pilot(Part):
    """ Part that steers to minimise the cross track error"""

    def __init__(self, pid, throttle):
        """
        Creating the PID
        :param PID pid:             PID controller part
        :param float throttle:      Fixed throttle value
        """
        super().__init__(pid=pid, throttle=throttle)
        self.pid = pid
        self.throttle = throttle

    def run(self, cte):
        """
        Donkey car parts interface. We return steering and throttle,
        where steering minimises the CTE using a PID controller and the
        throttle value is fixed in the constructor.

        :param float cte:   Current CTE value
        :return:            Tuple of (steering, throttle)
        :rtype:             2-tuple (float, float)
        """
        steer = self.pid.run(cte)
        logging.debug(f"CTE: {cte} steer: {steer}")
        return steer, self.throttle

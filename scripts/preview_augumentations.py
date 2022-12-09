#!/usr/bin/env python3
"""
Usage:
    preview_augmentations.py

Note:
    This script helps preview augmentations used when the model is being trained.
"""

import time
import cv2

from donkeycar.pipeline.augmentations import Augmentations


# Example augmentations
cropping = Augmentations.crop(0, 0, 100, 0, keep_size=True)
mask = Augmentations.trapezoidal_mask(10, 630, 100, 300, 50, 480)


def preview_augmentations():
    print('Connecting to Camera')
    capture = cv2.VideoCapture(0)
    time.sleep(2)
    if capture.isOpened():
        print('Camera Connected. Press "q" to stop the program - you need to '
              'focus on the running app to do so, not on the shell!')
    else:
        print('Unable to connect. Are you sure you are using the right camera '
              'parameters ?')
        return

    while True:
        success, frame = capture.read()
        if success:
            cropped = cropping(image=frame)['image']
            masked = mask(image=frame)['image']
            # Convert to RGB
            cv2.imshow('Preview', frame)
            cv2.imshow('Cropped', cropped)
            cv2.imshow('Trapezoidal Mask', masked)
            prompt = cv2.waitKey(1) & 0xFF
            if prompt == ord(' '):
                # Store output
                pass
            elif prompt == ord('q'):
                print('Bye bye...')
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    preview_augmentations()

import cv2
import numpy as np
import os


def framecapture(filename: str, path: str, name: str = "frame"):
    source = cv2.imread(filename)
    if not os.path.exists(path):
        os.mkdir(path)
    frameNum = 0
    while (True):
        success, frame = source.read()
        if success:
            cv2.imwrite(f'{path}{name}{frameNum}.jpg', frame)
        else:
            break
        frameNum = frameNum+150

    source.release() #why

def imgcap(filename: str, path: str):
    source = cv2.imread(filename)
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(f'{path}frame0.jpg', source)
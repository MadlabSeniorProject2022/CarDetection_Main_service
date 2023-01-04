import pickle
import cv2
import os

print(os.listdir())
COLOR_MODEL = pickle.load(open('./utils/model_22_version_clf_B.sav', 'rb'))

def colorPalette (img):
    resize = cv2.resize(img, (16, 16), interpolation = cv2.INTER_NEAREST)
    # Remove Border that may not car body
    resize = resize[2:15,2:15]
    resize = cv2.resize(resize, (8, 8), interpolation = cv2.INTER_CUBIC)
    resize = resize[1:7,1:7]
    resize = cv2.resize(resize, (4, 4), interpolation = cv2.INTER_AREA)
    resize = resize[1:3,1:3]
    return resize

def colorFeature(palette):
    result = []
    for line in palette:
        for color in line:
            result.extend(color)
    return result

def imgColorFeature (img):
    palette = colorPalette(img)
    print(palette)
    return colorFeature(palette)

def predict(img):
    feature = imgColorFeature(img)
    result = COLOR_MODEL.predict([feature])[0]
    return result

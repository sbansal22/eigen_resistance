"""
Pythons scipt to select parts of images and crop them to initialize the correct
size

Code slightly modified from:
https://docs.opencv.org/trunk/d4/dc6/tutorial_py_template_matching.html

"""
#from PIL import Image
import cv2
import numpy as np
import os

# Change 'test' and 'training' as needed to initialize

cwd = os.chdir('test')
files = os.listdir(cwd)

print(files)

for i in range(len(files)):
    print(files[i])
    cwd = os.chdir('../test')
    rgb = cv2.imread(files[i])
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('../resistor_template.png',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8

    loc = np.where( res >= threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1]+h)

    resistor = rgb[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    resistor = cv2.resize(resistor, (600, 250))
    cwd = os.chdir('../initialized_test')
    cv2.imwrite(str(i) + '.png',resistor)

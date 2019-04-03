"""
Pythons scipt to select parts of images and crop them to initialize the correct
size.
"""
#from PIL import Image
import cv2
import numpy as np
import os

# Change 'test' and 'training' as needed to initialize

cwd = os.chdir('new_test')
files = os.listdir(cwd)

print(files)

for i in range(len(files)):
    print(i)
    cwd = os.chdir('../new_test')
    rgb = cv2.imread(files[i])
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('../resistor_template.png',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.99

    loc = np.where( res >= threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1]+h)

    resistor = rgb[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    resistor = cv2.resize(resistor, (600, 250))
    cwd = os.chdir('../new_test_init')
    cv2.imwrite(str(i) + '.png',resistor)

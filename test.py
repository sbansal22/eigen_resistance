import os
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

template_image = 'resistor_template.png'
os.chdir('train')
files = os.listdir(os.getcwd())

rgb = cv2.imread('P2220545.JPG',0)
#gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('resistor_template.png',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(rgb, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where( res >= threshold)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
#for pt in zip(*loc[::-1]):
#    cv2.rectangle(rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
resistor = rgb[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
resistor = cv2.resize(resistor, (600,270))
os.chdir('../initialized_train')
cv2.imwrite(output,resistor)
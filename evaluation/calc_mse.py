import cv2
import numpy as np

def calc_mse(source, target, color_chart_area):
    source = np.reshape(source, [-1, 1]).astype(np.float32)
    target = np.reshape(target, [-1, 1]).astype(np.float32)
    mse = sum(np.power((source-target),2))
    return mse/((np.shape(source)[0]) - color_chart_area) # yes, color_chart_area x 3 makes more sense :-), we used this to evaluate all methods


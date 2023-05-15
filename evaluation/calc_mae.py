import cv2
import numpy as np

def calc_mae(source, target, color_chart_area):
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    source_norm = np.sqrt(np.sum(np.power(source,2),1))
    target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
    norm = source_norm * target_norm
    L = np.shape(norm)[0]
    inds = norm != 0
    angles = np.sum(source[inds, :] * target[inds, :],1)/norm[inds]
    angles[angles > 1] = 1
    f = np.arccos(angles)
    f[np.isnan(f)] = 0
    f = f * 180/np.pi
    return sum(f)/(L - color_chart_area)


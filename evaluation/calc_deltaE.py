import numpy as np
from skimage import color

def calc_deltaE(source, target, color_chart_area):
    # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source = color.rgb2lab(source/255)
    target = color.rgb2lab(target/255)
    # error = np.mean(color.deltaE_cie76(source,target))
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    delta_e = np.sqrt(np.sum(np.power(source - target,2),1))

    return sum(delta_e)/(np.shape(delta_e)[0] - color_chart_area)

#################################################
# References:
# [1] http://zschuessler.github.io/DeltaE/learn/
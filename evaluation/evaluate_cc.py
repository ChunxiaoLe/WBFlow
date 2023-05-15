from evaluation.calc_deltaE import calc_deltaE
from evaluation.calc_deltaE2000 import calc_deltaE2000
from evaluation.calc_mse import calc_mse
from evaluation.calc_mae import calc_mae

def evaluate_cc(corrected, gt, color_chart_area, opt=1):
    if opt == 1:
        return calc_deltaE2000(corrected,gt,color_chart_area)
    elif opt == 2:
         return calc_deltaE2000(corrected,gt,color_chart_area), calc_mse(corrected,gt,color_chart_area)
    elif opt == 3:
        return calc_deltaE2000(corrected,gt,color_chart_area), calc_mse(corrected,gt,color_chart_area), \
               calc_mae(corrected, gt, color_chart_area)
    elif opt == 4:
        return  calc_deltaE2000(corrected,gt,color_chart_area), calc_mse(corrected,gt,color_chart_area), \
                calc_mae(corrected, gt, color_chart_area), calc_deltaE(corrected,gt,color_chart_area)
    else:
        raise Exception('Error in evaluate_cc function')

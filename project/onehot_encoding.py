import numpy as np

def onehot_convert(label_str):
    comparison = ['AC', 'BC', 'BF', 'BL', 'BP', 'CF', 'GH', 'MA', 'ML', 'PP', 'SL', 'WO']
    result = np.zeros(len(comparison))
    result[comparison.index(label_str)] = 1

    return result
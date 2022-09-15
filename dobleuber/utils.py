"""
Author: Wbert Adrián Castro Vera (dobleuber)
Source: https://github.com/dobleuber/DeepReinforcementLearningUdacity
License: <unspecified>
"""

import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim

#!/usr/bin/env python3

import numpy as np
import pandas as pd

edats = [
    34, 48, 48, 55, 39, 39, 39, 35, 18, 54, 53, 29, 40, 37, 38,  23, 38,
    51, 71, 49, 46, 40, 39, 54, 48, 38, 34, 43, 39, 45, 42, 49,  32, 56,
    73, 57, 47, 54, 35, 29, 78, 63, 54, 59, 56, 71, 76, 80, 80,  78, 67,
    62, 52, 82, 67, 65, 83, 59, 79, 73, 57, 68, 81, 79, 87, 73, 108, 59,
    12, 15,  3, 26, 24, 16, 11,  2,  9, 23, 14, 12, 21, 27, 18,  19, 29,
    18, 13, 22, 13, 12, 28, 13, 19, 12, 13, 20, 14, 18
]

df = pd.DataFrame(edats, columns=['edat'])
percentils = df.quantile([0.20, 0.85])
print("Percentils:")
print(percentils)


quartils = df.quantile([0.25, 0.5, 0.75])
print("Quartils:")
print(quartils)

decils = df.quantile(np.arange(0.1, 1, 0.1))
print("Decils:")
print(decils)
#!/usr/bin/env python

import pandas as pd
import numpy as np

Y = pd.Series([50, 60, 70, 80])
pred_Y = pd.Series([52, 58, 68, 85])

mae = np.abs(Y - pred_Y).mean()
print(f'MAE with pandas: {mae:.2f}')

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(Y, pred_Y)
print(f'MAE with sklearn: {mae:.2f}')

mse = ((Y - pred_Y) ** 2).mean()
print(f'MSE with pandas: {mse:.2f}')

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y, pred_Y)
print(f'MSE with sklearn: {mse:.2f}')

rmse = np.sqrt(((Y - pred_Y) ** 2).mean())
print(f'RMSE with pandas: {rmse:.2f}')

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(Y, pred_Y))
print(f'RMSE with sklearn: {rmse:.2f}')

r2 = 1 - ((Y - pred_Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
print(f'R^2 with pandas: {r2:.2f}')

from sklearn.metrics import r2_score

r2 = r2_score(Y, pred_Y)
print(f'R^2 with sklearn: {r2:.2f}')

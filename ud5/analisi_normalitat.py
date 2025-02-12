import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('./../files/ud5/kung_san.csv', index_col=0)
df.info()

print(df.head())

# Pes de les dones majors a 15 anys
datos = df[(df.age > 15) & (df.male ==0)]
peso = datos['weight']

figure = plt.figure(figsize=(15, 5))
axes = figure.add_subplot()

axes.set_title('Distribución peso mujeres mayores de 15 años')
axes.set_xlabel('peso')
axes.set_ylabel('Densidad de probabilidad')
_ = sns.histplot(x=peso, axes=axes, kde=True, stat='density', bins=30)

# Histograma + curva normal teórica
# ==============================================================================

# Valores de la media (mu) y desviación típica (sigma) de los datos
mu = peso.mean()
sigma = peso.std()

# Valores teóricos de la normal en el rango observado
x_hat = np.linspace(min(peso), max(peso), num=100)
y_hat = stats.norm.pdf(x_hat, mu, sigma)

# Gráfico
figure = plt.figure(figsize=(15, 5))
axes = figure.add_subplot()

axes.set_title('Distribución peso mujeres mayores de 15 años')
axes.set_xlabel('peso')
axes.set_ylabel('Densidad de probabilidad')
_ = axes.plot(x_hat, y_hat, linewidth=2, label='normal', color="red")
_ = sns.histplot(x=peso, axes=axes, kde=True, stat='density', bins=30)
_ = axes.legend()

figure = plt.figure(figsize=(10, 5))
axes = figure.add_subplot()
#fit = True para estandarizar los datos, line = q para ajustar la línea a los cuartiles
sm.qqplot(peso, fit   = True, line  = "q", alpha = 0.4, ax    = axes)
axes.set_title('Gráfico Q-Q del peso mujeres mayores de 15 años', fontsize = 10,
             fontweight = "bold")
axes.tick_params(labelsize = 7)

print('Kursotis:', stats.kurtosis(peso))
print('Skewness:', stats.skew(peso))

# Shapiro-Wilk test
# ==============================================================================
print(f'Shapiro-Wilk test: ',stats.shapiro(peso))
print('Skew test:', stats.skewtest(peso))
print('Normal test:', stats.normaltest(peso))

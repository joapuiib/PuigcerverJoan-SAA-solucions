#!/usr/bin/env python3

import pandas as pd
from matplotlib.projections import projection_registry

"""
df = pd.DataFrame({"A":[12,  4,  5, 44,  1],
                   "B":[ 5,  2, 54,  3,  2],
                   "C":[20, 16,  7,  3,  8],
                   "D":[14,  3, 17,  3, 14]})
"""

df = pd.DataFrame({
    "Altura": [150, 160, 170, 180],
    "Pes": [50, 65, 71, 88],
    "Edat": [30, 40, 20, 25],
})

print("DataFrame:")
print(df)

# axis = 0 calculará la mitjana de cada columna (A, B, C, D)
column_mean = df.mean(axis = 0)
print("Mitjana de cada columna:")
print(column_mean)

# axis = 1 calculará la mitjana de cada fila (0, 1, 2, 3, 4)
row_mean = df.mean(axis = 1)
print("Mitjana de cada fila:")
print(row_mean)

mean = df.iloc[0].mean()
print("Mitjana de la primera fila:")
print(mean)

mean = df['A'].mean()
print("Mitjana de la columna A:")
print(mean)

median = df.median()
print("Mediana de cada columna:")
print(median)

mode = df.mode()
print("Moda de cada columna:")
print(mode)


variance = df.var()
print("Variança de cada columna:")
print(variance)

std = df.std()
print("Desviació estàndard de cada columna:")
print(std)

l = [20, 20, 50, 60 , 80]
s = pd.Series(l)
print(f"var de {l}: {s.var(ddof=0)}")
print(f"std de {l}: {s.std(ddof=0)}")

#!/usr/bin/env python3

from statistics import NormalDist
import numpy as np

alpha = 0.05
print("Nivell de significació:", alpha)
z_alpha = round(NormalDist().inv_cdf(alpha), 2)
print("Z-score de alpha:", z_alpha)

X = np.array([7, 6, 4, 3, 1, 9, 8, 7, 3, 1, 1, 0, 9, 4, 3, 2, 1, 5, 5, 6, 4])

# Mitjana i desviació estàndard de la mostra
mean = X.mean()
std = X.std()

# Z-score de la mitjana de la mostra
z = (mean - 5) / (std / np.sqrt(len(X)))
print()
print("Mitjana de la mostra:", mean)
print("Z-score de la mitjana de la mostra:", round(z, 2))

# Comprovació de la hipòtesi
p_value = NormalDist().cdf(z)
print()
print(f"Z-score de la mitjana de la mostra: {round(z, 2)} (p_value: {round(p_value, 4)})")
print(f"Z-score de alpha: {z_alpha} ({alpha})")
if z < z_alpha:
    print("Rebutgem la hipòtesi nul·la")
else:
    print("Acceptem la hipòtesi nul·la")


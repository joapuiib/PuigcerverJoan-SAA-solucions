#!/usr/bin/env python3

import pandas as pd

df = pd.DataFrame({
        "Altura": [150, 160, 170, 180],
        "Pes": [50, 65, 71, 88],
        "Edat": [30, 40, 60, 35],
})

print("DataFrame:")
print(df)
print()

cov = df.cov(ddof=0)
print("Matriu de covariància:")
print(cov)
print()

pearson = df.corr(method='pearson')
print("Correlació de Pearson:")
print(pearson)
print()

spearman = df.corr(method='spearman')
print("Correlació de Spearman:")
print(spearman)
print()

spearman = df.corr(method='kendall')
print("Correlació de Kendall:")
print(spearman)
print()

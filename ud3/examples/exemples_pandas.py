#!/usr/bin/env python

import pandas as pd

df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
print(df)

df['D'] = df['A'] * 2
print(df)

df.drop('D', axis=1, inplace=True)
print(df)

df.loc[0, 'A'] = 10
print(df)

fila = pd.DataFrame([[11, 12, 13]], columns=["A", "B", "C"], index=[3])
df = pd.concat([df, fila], axis=0)
print(df)

# Crear un DataFrame a partir d'un fitxer CSV
cotxes_df = pd.read_csv('../../files/ud3/cotxes.csv')
print("DataFrame a partir d'un fitxer CSV")
print(df)

# Filtrar els cotxes amb 'km' major que 70000
print("Cotxes amb més de 100000 km")
# print(cotxes_df['km'] > 70000)
print(cotxes_df.loc[cotxes_df['marca'] == 'Seat'])
print()

#!/usr/bin/env python3

import pandas as pd

# Distribució conjunta (totes les dades)
df_edats = pd.read_csv('../../files/ud2/generated_edats.csv', header=None, index_col=False, names=['edat'])
mean = df_edats.mean().values[0].round(2)
std = df_edats.std().values[0].round(2)

print("# Distribució conjunta")
print("Mitjana de les edats:", mean)
print("Desviació estàndard de les edats:", std)
print()

# Distribució mostral I (agafem 1 mostra de 10 dades)
sample1 = df_edats.sample(10, random_state=42)
mean = sample1.mean().values[0].round(2)
std = sample1.std().values[0].round(2)

print("Distribució mostral I")
print("Mitjana de la mostra 1:", mean)
print("Desviació estàndard de la mostra 1:", std)
print()

# Distribució mostral II (agafem 1 mostra de 100 dades)
sample2 = df_edats.sample(100, random_state=42)
mean = sample2.mean().values[0].round(2)
std = sample2.std().values[0].round(2)

print("Distribució mostral II")
print("Mitjana de la mostra 2:", mean)
print("Desviació estàndard de la mostra 2:", std)
print()

# Distribució mostral III (agafem 1 mostra de 1000 dades)
sample3 = df_edats.sample(1000, random_state=42)
mean = sample3.mean().values[0].round(2)
std = sample3.std().values[0].round(2)

print("Distribució mostral III")
print("Mitjana de la mostra 3:", mean)
print("Desviació estàndard de la mostra 3:", std)
print()

# Distribució mostral de la mitjana I (agafem 1 mostra de 10 dades)
sample1 = df_edats.sample(10, random_state=42)
mean = sample1.mean().values[0].round(2)
std = sample1.std().values[0].round(2)

print("Distribució mostral de la mitjana I")
print("Mitjana de la mostra 1:", mean)
print("Desviació estàndard de la mostra 1:", std)
print()

# Distribució mostral de la mitjana II (agafem 10 mostres de 10 dades)
means = []
stds = []
for i in range(10):
    sample = df_edats.sample(10, random_state=i)
    means.append(sample.mean().values[0])
    stds.append(sample.std().values[0])

mean = pd.Series(means).mean().round(2)
std = pd.Series(stds).mean().round(2)

print("Distribució mostral de la mitjana II")
print("Mitjana de les mitjanes:", mean)
print("Desviació estàndard de les mitjanes:", std)
print()

# Distribució mostral de la mitjana III (agafem 100 mostres de 10 dades)
means = []
stds = []
for i in range(100):
    sample = df_edats.sample(10, random_state=i)
    means.append(sample.mean().values[0])
    stds.append(sample.std().values[0])

mean = pd.Series(means).mean().round(2)
std = pd.Series(stds).mean().round(2)

print("Distribució mostral de la mitjana III")
print("Mitjana de les mitjanes:", mean)
print("Desviació estàndard de les mitjanes:", std)

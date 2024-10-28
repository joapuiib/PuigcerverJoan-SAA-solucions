import math
#!/usr/bin/env python3
from statistics import NormalDist
from scipy.stats import norm

"""
Ex1
Si la variable X se distribuye normalmente, ¿Cuál es la puntuación típica
que deja por encima el 57.53% de los sujetos?
"""
print("Ex1")
X = norm.ppf(1 - 0.5753)
print(f"X tal que P(X > x) = 0.5753: {X}")

"""
Ex2
En una variable con media de 4 y desviación típica de 2,
que se distribuye normalmente, ¿Qué proporción deja
por encima una puntuación directa de 2.5?
"""
print("Ex2")
dist = NormalDist(mu=4, sigma=2)
p = 1 - dist.cdf(2.5)
print(f"Probabilitat de ser més de 2.5: {p}")

"""
Ex3
Las alturas, expresadas en cm, de un colectivo de 300 estudiantes
se distribuye normalmente con una media de 160 cm y una desviación típica de 20.
¿Cuántos estudiantes son los que miden más de 140 cm y menos de 180 cm?
"""
print("Ex3")
dist = NormalDist(mu=160, sigma=20)
p = dist.cdf(180) - dist.cdf(140)
n = 300 * p
print(f"Estudiants que miden més de 140cm i menys de 180cm: {n}")

"""
Ex4
Supuesta una distribución normal obtenida con 600 personas,
la puntuación directa 20 deja por debajo de sí 114 personas.
Sabiendo que la varianza vale 16, ¿Cuál es el número de personas
que obtienen puntuaciones mayores que 24 y menores que 28?
"""
print("Ex4")
std = 16
p_menor_20 = 114.0 / 600.0
z = norm.ppf(p_menor_20)
mu = 20 - z * std
print(f"Mitjana: {mu}")

# Distribucio normal
dist = NormalDist(mu=mu, sigma=16)
p = dist.cdf(28) - dist.cdf(24)
print(f"[NORM] Persones que obtenen puntuacions majors que 24 i menors que 28: {p}")

# Distribucio normal estandar
dist = NormalDist(mu=0, sigma=1)
z_28 = (28 - mu) / 16
z_24 = (24 - mu) / 16
p = dist.cdf(z_28) - dist.cdf(z_24)
print(f"[STD NORM] Persones que obtenen puntuacions majors que 24 i menors que 28: {p}")

"""
Ex5
# Las puntuaciones de n personas en un test de inteligencia se distribuyen
normalmente con media 100 y varianza 25. Sabemos además 1587 de ellas
obtuvieron puntuaciones superiores a 105. Calcula el valor de n0
"""
print("Ex5")
std = math.sqrt(25)
dist = NormalDist(mu=100, sigma=std)
p_greater_105 = 1 - dist.cdf(105)
print(f"Probabilitat de ser més de 105: {p_greater_105}")
N = 1587.0 / p_greater_105
print(f"N: {N}")
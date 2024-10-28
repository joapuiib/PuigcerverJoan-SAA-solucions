#!/usr/bin/env python3

from statistics import NormalDist
from scipy.stats import norm

height_dist = NormalDist(mu=165, sigma=15)

p_between_150_165 = height_dist.cdf(165) - height_dist.cdf(150)
print(f"Probabilitat de ser entre 150cm i 165cm: {p_between_150_165}")

p_less_150 = height_dist.cdf(150)
print(f"Probabilitat de ser menys de 150cm: {p_less_150}")

p_more_150 = 1 - height_dist.cdf(150)
print(f"Probabilitat de ser més de 150cm: {p_more_150}")

X = norm.ppf(0.4247)
print(f"X tal que P(X < x) = 0.4247: {X}")
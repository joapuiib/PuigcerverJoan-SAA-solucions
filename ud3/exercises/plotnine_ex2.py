#!/usr/bin/env python

from plotnine import *
import pandas as pd

df = pd.read_csv('../../files/ud3/athlete_events.csv')
# Filtrar les dades per a que només apareguen els atletes de la Xina
df = df[df['NOC'] == 'RUS']
# Filtrar les dades per a que només apareguen les medalles
df.dropna(subset=['Medal'], inplace=True)
print(df.head())


"""
# Exercici 2.a
Crea un gràfic de barres apilades que mostre la distribució de les medalles
guanyades pels atletes de Rússia per any i gènere.

Crea els gràfics separats en dues columnes.
"""
plot = (
    ggplot(df)
    + aes(x='Year', fill='Sex')
    + geom_bar(stat='count')
    + facet_wrap('~Sex')
)

"""
# Exercici 2.b
Modifica el gràfic anterior per mostrar els gràfics en 2 files separades.
"""

plot = (
    ggplot(df)
    + aes(x='Year', fill='Sex')
    + geom_bar(stat='count')
    + facet_wrap('~Sex', nrow=2)
)

"""
# Exercici 2.c
Modifica el gràfic anterior per mostrar les columnes de medallistes homes i
dones en el mateix gràfic, un al costat de l'altre.
"""

plot = (
    ggplot(df)
    + aes(x='Year', fill='Sex')
    + geom_bar(stat='count', position='dodge')
)


"""
# Exercici 2.d
Elimina les medalles guanyades abans de 1994 del gràfic anterior.
"""

df = df[df['Year'] >= 1994]

plot = (
    ggplot(df)
    + aes(x='Year', fill='Sex')
    + geom_bar(stat='count', position='dodge')
)


"""
# Exercici 2.e
Fes les modificacions necessàries perquè el gràfic de l'exercici anterior quede
semblant a aquest.
"""

df = df[df['Year'] >= 1994]

plot = (
    ggplot(df)
    + aes(x='Year', fill='Sex')
    + geom_bar(stat='count', position='dodge')
    + labs(
        title='Medalles dels atletes de Rússia per any i gènere',
        x='Any',
        y='Nombre de medalles',
        fill='Gènere'
    )
    + scale_x_continuous(
        breaks=sorted(df['Year'].unique()),
    )
    + scale_fill_manual(
        values={
            'M': '#93c47d',
            'F': '#8e7cc3'
        }
    )
)

plot.show()

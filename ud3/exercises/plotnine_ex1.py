#!/usr/bin/env python

from plotnine import *
import pandas as pd
from pandas.api.types import CategoricalDtype

df = pd.read_csv('../../files/ud3/athlete_events.csv')
# Filtrar les dades per a que només apareguen els atletes de la Xina
df = df[df['NOC'] == 'CHN']
# Filtrar les dades per a que només apareguen les medalles
df.dropna(subset=['Medal'], inplace=True)
print(df.head())



"""
# Exercici 1.a
Crea un gràfic de barres que mostre el nombre de medalles
guanyades pels atletes de la Xina en cada esport.
"""

plot = (
    ggplot(df)
    + aes(x='Sport')
    + geom_bar()
)

"""
# Exercici 1.b
Modifica el gràfic anterior perquè es diferencien les medalles
canviant el color segons el tipus.
"""

plot = (
    ggplot(df)
    + aes(x='Sport', fill='Medal')
    + geom_bar()
)

"""
# Exercici 1.c
Fes les modificacions necessàries perquè el gràfic de
l'exercici anterior quede semblant a aquest
"""

# Convert 'Medal' to a categorical variable with specified order
medal_order = CategoricalDtype(categories=['Gold', 'Silver', 'Bronze'], ordered=True)
df['Medal'] = df['Medal'].astype(medal_order)

plot = (
    ggplot(df)
    + aes(x='Sport', fill='Medal')
    + geom_bar(position=position_stack(reverse=True))
    + theme(axis_text_x=element_text(rotation=45, hjust=1))
    + scale_fill_manual(
        values={
            'Gold': '#FFD700',
            'Silver': '#C0C0C0',
            'Bronze': '#CD7F32'
        }
    )
    + labs(title='Medalles guanyades pels atletes de la Xina en cada esport',
           x='Esport',
           y='Nombre de medalles'
    )
)


"""
Ordenar els esport segons el nombre de medalles guanyades
"""

medal_order = CategoricalDtype(categories=['Gold', 'Silver', 'Bronze'], ordered=True)
df['Medal'] = df['Medal'].astype(medal_order)

sport_order = CategoricalDtype(
    categories=df.groupby('Sport').agg(
        Gold = ('Medal', lambda x: (x == 'Gold').sum()),
        Silver = ('Medal', lambda x: (x == 'Silver').sum()),
        Bronze = ('Medal', lambda x: (x == 'Bronze').sum())
    ).sort_values(
        by=['Bronze', 'Gold', 'Silver'],
        ascending=False
    ).index,
    ordered=True
)
df['Sport'] = df['Sport'].astype(sport_order)


plot = (
    ggplot(df)
    + aes(x='Sport', fill='Medal')
    + geom_bar(position=position_stack(reverse=True))
    + theme(axis_text_x=element_text(rotation=45, hjust=1))
    + scale_fill_manual(
        values={
            'Gold': '#FFD700',
            'Silver': '#C0C0C0',
            'Bronze': '#CD7F32'
        }
    )
    + labs(title='Medalles guanyades pels atletes de la Xina en cada esport',
           x='Esport',
           y='Nombre de medalles'
    )
)

plot.show()

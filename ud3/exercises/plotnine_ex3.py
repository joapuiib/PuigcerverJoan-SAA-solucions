#!/usr/bin/env python

from plotnine import *
import pandas as pd

df = pd.read_csv('../../files/ud3/athlete_events.csv')

"""
# Exercici 3.a
Crea un gràfic de línies que mostre com ha evolucionat la quantitat de
participants als Jocs Olímpics al llarg dels anys similar al següent:
"""
plot = (
    ggplot(df)
    + aes(x='Year')
    + geom_line(stat='count', color='red')
    + scale_x_continuous(breaks=range(df['Year'].min(), df['Year'].max() + 1, 4))
    + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(15, 6))
    + geom_point(stat='count', color='blue')
    + geom_text(
        aes(label=after_stat('count')),
        stat='count',
        va='bottom',
        size=8,
        nudge_y=100
    )
    + labs(title='Evolució de la Participació per Any', x='Any', y='Nombre de Participants')
)

#plot.show()

"""
# Exercici 3.a
Arregla les dades perquè mostren dades amb més rellevància estadística.
"""

df = df[df['Season'] == 'Summer']

plot = (
    ggplot(df)
    + aes(x='Year')
    + geom_line(stat='count', color='red')
    + scale_x_continuous(breaks=range(df['Year'].min(), df['Year'].max() + 1, 4))
    + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(15, 6))
    + geom_point(stat='count', color='blue')
    + geom_text(
        aes(label=after_stat('count')),
        stat='count',
        va='bottom',
        size=8,
        nudge_y=100
    )
    + labs(title='Evolució de la Participació per Any', x='Any', y='Nombre de Participants')
)

plot.show()

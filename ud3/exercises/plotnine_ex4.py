#!/usr/bin/env python

from plotnine import *
import pandas as pd

df = pd.read_csv('../../files/ud3/athlete_events.csv')

"""
# Exercici 4
Tria 5 esports i mostra la evolució de la participació femenina al llarg dels anys amb un gràfic:
"""
esports = ['Athletics', 'Swimming', 'Gymnastics', 'Art Competitions']
df = df[df['Sex'] == 'F']
df = df[df['Sport'].isin(esports)]

# Manually compute the counts of participants per year and sport
df_counts = df.groupby(['Year', 'Sport']).size().reset_index(name='count')

# Filter the original df to get the last year for each sport
df_last_year = df_counts.loc[df_counts.groupby('Sport')['Year'].idxmax()]

plot = (
    ggplot(df_counts)  # Use the df_counts that has Year, Sport, and count
    + aes(x='Year', y='count', color='Sport')
    + geom_line()  # Plot lines based on the computed counts
    + scale_x_continuous(breaks=range(df['Year'].min(), df['Year'].max() + 1, 4))
    + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(15, 6))
    + labs(title='Evolució de la Participació Femenina per Any', x='Any', y='Nombre de Participants')
    # Add sport names at the last year for each sport
    + geom_text(
        aes(label='Sport'),  # Show sport names as labels
        data=df_last_year,   # Use only the last year data for each sport
        va='bottom',
        ha='right',
        nudge_x=0.5,  # Slight horizontal nudge for better readability
        size=8,
        show_legend=False
    )
)

plot.show()

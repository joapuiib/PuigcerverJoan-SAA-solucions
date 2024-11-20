#!/usr/bin/env python

from plotnine import *
from plotnine.data import mtcars

print("Dades `mtcars`:")
print(mtcars)

plot = (
    ggplot(mtcars)
    + aes("wt", "mpg", color="factor(gear)")
    + geom_point()
    + labs(title="Consum (wt) vs Pes (mpg)",
           x="Pes en lliures",
           y="Consum en milles per galó",
           color="Nombre de marxes"
    )
    + facet_wrap("cyl")
    + theme_seaborn()
)

plot.show()

exit()

line_plot = (
    ggplot(mtcars)
    + aes(x='wt', y='mpg', color='factor(gear)')
    + geom_line()
)

line_plot.show()

bar_plot = (
    ggplot(mtcars)
    + aes(x='factor(cyl)', fill='factor(gear)')
    + geom_bar()
)

bar_plot.show()
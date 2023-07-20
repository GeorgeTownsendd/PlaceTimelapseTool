from PlaceMapTools import *

#canvas = Canvas('place_2022')
#canvas.add_subcanvas('00')
#canvas.subcanvases['00'].display()

#current = canvas.subcanvases['00']


canvas = Canvas('place_2022', datetime(2022, 4, 2, 23, 46))

example_section = Section(canvas, (950, 950), 100, 50)

example_section.display()
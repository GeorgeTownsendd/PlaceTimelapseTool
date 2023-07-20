from PlaceMapTools import *

#canvas = Canvas('place_2022')
#canvas.add_subcanvas('00')
#canvas.subcanvases['00'].display()
#current = canvas.subcanvases['00']


#canvas = Canvas('place_2022', datetime(2022, 4, 2, 23))
#reference_section = ReferenceSection(canvas, 'HI_test')
#reference_section.display_section_plot()

from datetime import datetime

canvas = Canvas('place_2022', datetime(2022, 4, 2, 23))
reference_section = ReferenceSection(canvas, 'HI_test')

start_time = datetime(2022, 4, 2, 23, 0, 0)
end_time = datetime(2022, 4, 2, 23, 20, 0)
reference_section.create_timelapse(start_time, end_time)


#example_section = Section(canvas, (48, 777), 42, 54)
#example_section.display_section_plot()
#canvas.create_reference_section('HI_test', (48, 777), 42, 54)
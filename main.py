from PlaceMapTools import *

#canvas = Canvas('place_2022')
#canvas.add_subcanvas('00')
#canvas.subcanvases['00'].display()
#current = canvas.subcanvases['00']


#canvas = Canvas('place_2022', datetime(2022, 4, 2, 23))
#reference_section = ReferenceSection(canvas, 'HI_test')
#reference_section.display_section_plot()

canvas = Canvas('place_2022', datetime(2022, 4, 3, 1))
reference_section = ReferenceSection(canvas, 'HI_test')

fig, axs = reference_section.create_three_part_plot()
plt.show()



#example_section = Section(canvas, (48, 777), 42, 54)
#example_section.display_section_plot()
#canvas.create_reference_section('HI_test', (48, 777), 42, 54)
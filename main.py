from canvas_analysis import *


if __name__ == "__main__":
    event = Event("place_2023")
    #x = event.generate_change_history_image('NG 15x15 + HI 11x11 + 11x6FF + other things + robot', timedelta(hours=3))
    #start_time = datetime.utcnow() - timedelta(hours=3)
    #end_time = datetime.utcnow()
    #ref_section = ReferenceSection.download_reference_sections('https://rentry.co/HITemplate/raw', 'place_2023')

    #event.create_basic_timelapse('NG 15x15 + HI 11x11 + 11x6FF + other things + robot', start_time=start_time, end_time=end_time)



    refname = 'TaskBar'
    #event = Event("place_2023")
    top_left = Coordinate(687, 971, 'reddit')
    event.canvas_images[-1].add_reference_section(refname, top_left, 40, 40)

    #start_time = datetime.utcnow() - timedelta(hours=17.75)
    #end_time = datetime.utcnow() - timedelta(hours=16)

    start_time = datetime.utcnow() - timedelta(hours=3)
    end_time = datetime.utcnow()# - timedelta(hours=16)

    event.create_basic_timelapse(coordinates_from=refname, start_time=start_time, end_time=end_time)
    #top_left_reddit = (-500, 178)
    #top_left_reddit = (215, 125)
    #top_left = Coordinate(215, 115, 'reddit')#reddit_to_image_coordinates(top_left_reddit)

    #ref_section = ReferenceSection.download_reference_section('https://brown.ee/LMoP1Zmv.json', 'place_2023')
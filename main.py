import PIL
import requests
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
from duplicatefilechecker import check_for_duplicates
import shutil
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from dateutil import tz

localtz = tz.tzlocal()
utc = tz.gettz('UTC')



image_dir = 'images/'
list_of_image_master = sorted(['images/' + d for d in os.listdir(image_dir)], key=lambda x: datetime.strptime(x[7:-4], '%Y-%m-%d %H:%M:%S.%f'))

hi_coordinates = ((48,777),(110,830))


def save_current_image(image_list_file='imagelist.txt'):
    image_list = open(image_list_file, 'a')
    start_time = datetime.utcnow()

    r = requests.get("https://canvas.codes/canvas")
    finish_time = datetime.utcnow()
    download_time = finish_time - start_time
    print(download_time)

    if r.status_code == 200:
        page = r.text
        image_url = page[page.index('https://'):page.index('.png')+4]
        img_data = requests.get(image_url).content
        image_filename = 'images/' + str(start_time) + '.png'

        with open(image_filename, 'wb') as handler:
            handler.write(img_data)

        image_list.write(image_filename)
        image_list.close()

        return image_filename
    else:
        return False


def load_image_as_np(image_filename=str('images/' + '2022-04-02 08:39:52.140427.png')):
    try:
        image_obj = Image.open(image_filename).convert('RGB')

        return np.array(image_obj)
    except PIL.UnidentifiedImageError:
        print('cannot identify image file')


def select_area(image_array, section, subsection='full'):
    bl, tr = section.sections[subsection]
    image_truth_area = image_array[bl[1]:tr[1],bl[0]:tr[0]]

    return image_truth_area


def get_differences(image1, image2, section, subsection='full'):
    #print(image1, image2)

    image1 = load_image_as_np(image1)
    image2 = load_image_as_np(image2)

    #print(image1.shape, image2.shape)

    image1_truth_area = select_area(image1, section=section, subsection=subsection)
    image2_truth_area = select_area(image2, section=section, subsection=subsection)



    identical_pixels = []
    for image_r, truth_r in zip(image1_truth_area, image2_truth_area):
        p_row = []
        for ip, tp in zip(image_r, truth_r):
            p_row.append((ip == tp).all())
        identical_pixels.append(p_row)

    return np.array(identical_pixels)


def get_image_diff(image1, image2, section, subsection='full'):
    image1_img = load_image_as_np(image1)
    image2_img = load_image_as_np(image2)
    identical_pixels_mask = get_differences(image1, image2, section=section, subsection=subsection)
    changed_pixels_mask = np.invert(identical_pixels_mask)

    y = np.expand_dims(changed_pixels_mask,axis=2)
    newmask = np.concatenate((y,y,y),axis=2)

    changedpixels = newmask * select_area(image2_img, section=section, subsection=subsection)

    before_image = select_area(image1_img, section=section, subsection=subsection)
    current_image = select_area(image2_img, section=section, subsection=subsection)
    diff_image = changedpixels

    return before_image, current_image, diff_image


def plot_difference(image1, image2, section, subsection='full', imagename='example.png', show=True):
    fig, axs = plt.subplots(1, 3)
    before_ax, current_ax, diff_ax = axs

    image1_img = load_image_as_np(image1)
    image2_img = load_image_as_np(image2)

    identical_pixels_mask = get_differences(image1, image2, section=section, subsection=subsection)
    changed_pixels_mask = np.invert(identical_pixels_mask)

    y = np.expand_dims(changed_pixels_mask,axis=2)
    newmask = np.concatenate((y,y,y),axis=2)

    changedpixels = newmask * select_area(image2_img, section=section, subsection=subsection)

    before_image = select_area(image1_img, section=section, subsection=subsection)
    before_ax.imshow(before_image)
    before_ax.set_title(image1[12:-11])

    current_image = select_area(image2_img, section=section, subsection=subsection)
    current_ax.imshow(current_image)
    current_ax.set_title(image2[12:-11])

    diff_image = changedpixels
    diff_ax.imshow(diff_image)
    diff_ax.set_title('Changed Pixels')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(imagename)

    return newmask


def remove_duplicate_images(directory='images/', trash_directory='/home/george/Scripts/place_activity_detector/duplicates/'):
    duplicates = check_for_duplicates([directory])

    for duplicate_file in duplicates:
        shutil.copy(duplicate_file, trash_directory)
        shutil.remove(duplicate_file)

    return duplicates


def get_list_of_images(image_dir='images/'):
    global list_of_image_master
    return list_of_image_master
    #return #sorted(['images/' + d for d in os.listdir(image_dir)], key=lambda x: datetime.strptime(x[7:-4], '%Y-%m-%d %H:%M:%S.%f'))


def save_periodically():
    n = len([x for x in os.listdir('images/') if x.endswith('.png')])
    while True:
        n += 1
        save_current_image()
        print('Saved image {}: {}'.format(n, str(datetime.utcnow())))
        time.sleep(5)


def count_different_pixels(image1, image2, section, subsection='full'):
    identical_pixels_mask = get_differences(image1, image2, section, subsection=subsection)
    print(identical_pixels_mask)
    changed_pixels_mask = np.invert(identical_pixels_mask)

    return changed_pixels_mask.sum()


def animate_difference_overview(start, end, section, subsection='full', skip=1, truth='standard'):
    l = get_list_of_images()[start:end][::skip]

    if truth == 'standard':
        image_standard = section.reference_image_file
    elif truth == 'original':
        image_standard = l[0]
    elif truth == 'final':
        image_standard = l[-1]

    image_standard = '/home/george/Scripts/place_activity_detector/images/2022-04-04 12:03:54.762193.png'

    times = [datetime.strptime(x[7:-4], '%Y-%m-%d %H:%M:%S.%f') for x in l]

    from_original = []
    differences = []
    timestamps = []
    last_image = l[0]
    last_time = times[0]
    used_images = []

    for current_image, current_time in zip(l[1:], times[1:]):
        changed_pixels_per_second = count_different_pixels(last_image, current_image, section, subsection=subsection) / int((current_time - last_time).seconds)
        used_images.append((image_standard, current_image))
        from_original.append(count_different_pixels(image_standard, current_image, section, subsection=subsection))
        differences.append(changed_pixels_per_second)
        timestamps.append(current_time)
        last_image = current_image
        last_time = current_time

    n = 1
    #original_image = load_image_as_np(original)
    for standard, current_image in used_images:
        animate_frame(image_standard, current_image, timestamps[:n], differences[:n], from_original[:n], section, subsection=subsection, frame_title=str(n), end_point_diff=(times[-1], max(differences)), end_point_incorrect=(times[-1], max(from_original)), truth=truth)
        print(n)
        n += 1


def animate_frame(image1, image2, timestamps, differences, from_original, section, subsection='full', frame_title='example', end_point_diff='', end_point_incorrect='', savefig=True, truth='standard'):
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    #spec = fig.add_gridspec(3, 3)
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, height_ratios=[3, 1, 1])

    ax01 = fig.add_subplot(spec[0, 0])
    ax02 = fig.add_subplot(spec[0, 1])
    ax03 = fig.add_subplot(spec[0, 2])

    ax10 = fig.add_subplot(spec[1, :])

    ax11 = fig.add_subplot(spec[2, :])

    for a in [ax01, ax02, ax03]:
        a.set_xticks([])
        a.set_yticks([])

    pixel_per_second_axis, incorrect_pixel_axis = ax10, ax11
    pixel_per_second_axis.set_title('Pixel per second change')
    incorrect_pixel_axis.set_title('Number of different pixels')

    before, current, diff = get_image_diff(image1, image2, section, subsection=subsection)
    ax01.imshow(before)
    ax01.set_title('Final')
    ax02.imshow(current)

    ax02.set_title(image2[12:-11])
    ax03.imshow(diff)
    ax03.set_title('Difference')

    pixel_per_second_axis.plot([timestamps[0]], [0])
    incorrect_pixel_axis.plot([timestamps[0]], [0])
    if end_point_diff:
        pixel_per_second_axis.plot([end_point_diff[0]], [end_point_diff[1]])
        incorrect_pixel_axis.plot([end_point_incorrect[0]], [end_point_incorrect[1]])
    pixel_per_second_axis.plot(timestamps, differences)
    incorrect_pixel_axis.plot(timestamps, from_original, color='orange')

    pixel_per_second_axis.set_xticklabels(pixel_per_second_axis.get_xticklabels())#, rotation=45)
    incorrect_pixel_axis.set_xticklabels(incorrect_pixel_axis.get_xticklabels())#, rotation=45)

    date_form = DateFormatter('%H:%M')
    pixel_per_second_axis.xaxis.set_major_formatter(date_form)
    incorrect_pixel_axis.xaxis.set_major_formatter(date_form)

    if savefig:
        plt.savefig('timelapse/' + frame_title + '.png')
        plt.close('all')


def generate_timelapse_images(start, end, section, subsection='full'):
    l = get_list_of_images()[start:end]

    for n, i in enumerate(l):
        t = (l[0], i)

        plot_difference(*t, imagename='timelapse/{}.png'.format(n), section=section, subsection=subsection, show=False)
        print(n)
        plt.close('all')


class Section:
    def __init__(self, name):
        references = os.listdir('references/')
        if name + '.png' in references and name + '.txt' in references:
            self.reference_image_file = 'references/' + name + '.png'
            self.reference_image = load_image_as_np(self.reference_image_file)
            with open('references/' + name + '.txt', 'r') as f:
                lines = [l[:-1] for l in f.readlines()]
            sections = [lines[(n*3):(n+1)*3] for n in range(len(lines)//3)]
            sectiondic = {name : ([int(n) for n in p1.split(',')], [int(n) for n in p2.split(',')]) for name, p1, p2 in sections}

            self.sections = sectiondic
        else:
            print('Unable to find reference ', name)

    def select_section_from_image(self, sectionname, image_name='latest'):
        if image_name == 'latest':
            l = get_list_of_images()

            image = load_image_as_np(l[-1])
        else:
            image = load_image_as_np(image_name)

        selected_area = select_area(image, self.sections[sectionname])
        return selected_area

    def display_section(self, sectionname):
        selected_area = self.select_section_from_image(sectionname)

        plt.imshow(selected_area)
        plt.show()


def graph_section_activity(image1, image2, section):
    section_names = section.sections.keys()
    section_activity = {}
    for sect in section_names:
        if sect not in ['romartface', 'caunlogo', 'watermelon', 'redlogo']:
            section_activity[sect] = count_different_pixels(image1, image2, section, subsection=sect)
            print(sect)

    return section_activity

    #save_periodically()

l = get_list_of_images()

hellointernet = Section('hellointernet_2')
#x = graph_section_activity(l[-2], l[-1], section=hellointernet)

animate_difference_overview(-14100, -13100, hellointernet, subsection='full', skip=2, truth='standard')


#generate_timelapse_images(2450, 2600, section=hellointernet, subsection='audrey')
